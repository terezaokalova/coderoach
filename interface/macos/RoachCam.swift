import AVFoundation
import AppKit
import CoreImage
import Foundation
import ImageIO
import Network
import UniformTypeIdentifiers

final class Server {
    private let queue = DispatchQueue(label: "roachcam.http")
    private var listener: NWListener?
    private var latest = Data()
    private let lock = NSLock()

    func update(_ jpeg: Data) {
        lock.lock()
        latest = jpeg
        lock.unlock()
    }

    func start(port: UInt16) {
        let params = NWParameters.tcp
        listener = try? NWListener(using: params, on: NWEndpoint.Port(rawValue: port)!)
        listener?.newConnectionHandler = { [weak self] conn in
            conn.start(queue: self?.queue ?? .main)
            self?.serve(conn)
        }
        listener?.start(queue: queue)
    }

    private func serve(_ conn: NWConnection) {
        conn.receive(minimumIncompleteLength: 1, maximumLength: 4096) { [weak self] _, _, _, _ in
            guard let self else { return }
            self.lock.lock()
            let jpeg = self.latest
            self.lock.unlock()
            var body = Data()
            body.append(contentsOf: Array("HTTP/1.1 200 OK\r\nContent-Type: image/jpeg\r\nContent-Length: \(jpeg.count)\r\nConnection: close\r\n\r\n".utf8))
            body.append(jpeg)
            conn.send(content: body, completion: .contentProcessed { _ in
                conn.cancel()
            })
        }
    }
}

final class App: NSObject, NSApplicationDelegate, AVCaptureVideoDataOutputSampleBufferDelegate {
    let window = NSWindow(
        contentRect: NSRect(x: 120, y: 120, width: 960, height: 640),
        styleMask: [.titled, .closable, .resizable, .miniaturizable],
        backing: .buffered,
        defer: false
    )
    let session = AVCaptureSession()
    let output = AVCaptureVideoDataOutput()
    let preview = AVCaptureVideoPreviewLayer()
    let server = Server()
    let context = CIContext(options: [.useSoftwareRenderer: false])
    let videoQueue = DispatchQueue(label: "roachcam.video")
    var lastWrite = Date.distantPast

    func applicationDidFinishLaunching(_ notification: Notification) {
        server.start(port: 8765)
        requestCamera()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }

    private func requestCamera() {
        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                if granted {
                    self.startSession()
                } else {
                    self.window.title = "Camera denied — allow RoachCam in Privacy settings"
                }
            }
        }
    }

    private func devices() -> [AVCaptureDevice] {
        var types: [AVCaptureDevice.DeviceType] = [.builtInWideAngleCamera, .external]
        types.append(.continuityCamera)
        let session = AVCaptureDevice.DiscoverySession(
            deviceTypes: types,
            mediaType: .video,
            position: .unspecified
        )
        return session.devices
    }

    private func startSession() {
        let found = devices()
        FileHandle.standardError.write(Data("devices: \(found.map(\.localizedName))\n".utf8))
        let device = found.first { $0.localizedName.localizedCaseInsensitiveContains("iphone") } ?? found.last
        guard let device else {
            window.title = "No camera found"
            return
        }
        do {
            session.beginConfiguration()
            session.sessionPreset = .high
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) { session.addInput(input) }
            output.alwaysDiscardsLateVideoFrames = true
            output.setSampleBufferDelegate(self, queue: videoQueue)
            if session.canAddOutput(output) { session.addOutput(output) }
            session.commitConfiguration()
            session.startRunning()
            window.title = "iPhone livestream — \(device.localizedName)"
            FileHandle.standardError.write(Data("using \(device.localizedName)\n".utf8))
        } catch {
            window.title = "Failed to open \(device.localizedName)"
        }
    }

    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        let now = Date()
        guard now.timeIntervalSince(lastWrite) > 0.12 else { return }
        lastWrite = now
        guard let pixel = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        let image = CIImage(cvImageBuffer: pixel)
        guard let cg = context.createCGImage(image, from: image.extent) else { return }
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(
            data, UTType.jpeg.identifier as CFString, 1, nil
        ) else { return }
        CGImageDestinationAddImage(dest, cg, [kCGImageDestinationLossyCompressionQuality: 0.7] as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { return }
        let jpeg = data as Data
        server.update(jpeg)
        try? jpeg.write(to: URL(fileURLWithPath: "/tmp/roach_cam/live.jpg"), options: .atomic)
    }
}

mkdirp()
let app = NSApplication.shared
let delegate = App()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()

func mkdirp() {
    try? FileManager.default.createDirectory(
        atPath: "/tmp/roach_cam",
        withIntermediateDirectories: true
    )
}
