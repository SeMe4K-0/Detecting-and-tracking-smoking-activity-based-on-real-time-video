import AVFoundation
import CoreVideo

protocol CameraManagerDelegate: AnyObject {
    func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer)
    func cameraManager(_ manager: CameraManager, didSwitchTo position: AVCaptureDevice.Position)
}

final class CameraManager: NSObject {
    weak var delegate: CameraManagerDelegate?

    private let session = AVCaptureSession()
    private let sessionQueue = DispatchQueue(label: "camera.session.queue")
    private var videoOutput: AVCaptureVideoDataOutput?

    let previewLayer: AVCaptureVideoPreviewLayer

    private var isConfigured = false
    private var currentPosition: AVCaptureDevice.Position = .back
    private var isSwitching = false

    override init() {
        previewLayer = AVCaptureVideoPreviewLayer(session: session)
        previewLayer.videoGravity = .resizeAspectFill
        super.init()
    }

    var cameraPosition: AVCaptureDevice.Position {
        return currentPosition
    }

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized:
            sessionQueue.async { [weak self] in
                self?.configureSessionIfNeeded()
                self?.session.startRunning()
            }
        case .notDetermined:
            AVCaptureDevice.requestAccess(for: .video) { [weak self] granted in
                if granted {
                    self?.sessionQueue.async {
                        self?.configureSessionIfNeeded()
                        self?.session.startRunning()
                    }
                } else {
                    print("Camera access denied")
                }
            }
        default:
            print("Camera access denied or restricted")
        }
    }

    func stop() {
        sessionQueue.async { [weak self] in
            self?.session.stopRunning()
        }
    }

    func switchCamera() {
        sessionQueue.async { [weak self] in
            guard let self = self else { return }
            guard !self.isSwitching else { return }
            guard self.session.isRunning else {
                print("Cannot switch: session not running")
                return
            }
            self.isSwitching = true
            defer { self.isSwitching = false }

            let newPosition: AVCaptureDevice.Position = (self.currentPosition == .back) ? .front : .back

            self.session.beginConfiguration()

            // Remove existing video inputs
            for input in self.session.inputs {
                if input is AVCaptureDeviceInput {
                    self.session.removeInput(input)
                }
            }

            guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: newPosition) else {
                print("Camera not available for position \(newPosition)")
                self.session.commitConfiguration()
                return
            }

            do {
                let input = try AVCaptureDeviceInput(device: device)
                if self.session.canAddInput(input) {
                    self.session.addInput(input)
                    self.currentPosition = newPosition
                } else {
                    print("Cannot add input for position \(newPosition)")
                    self.session.commitConfiguration()
                    return
                }
            } catch {
                print("Cannot create camera input: \(error)")
                self.session.commitConfiguration()
                return
            }

            self.session.commitConfiguration()

            // Update output connection after configuration
            if let connection = self.videoOutput?.connection(with: .video) {
                connection.videoOrientation = .portrait
                connection.isVideoMirrored = (newPosition == .front)
            }

            DispatchQueue.main.async { [weak self] in
                guard let self = self else { return }
                self.delegate?.cameraManager(self, didSwitchTo: newPosition)
            }
        }
    }

    private func configureSessionIfNeeded() {
        guard !isConfigured else { return }
        isConfigured = true
        session.beginConfiguration()
        defer { session.commitConfiguration() }

        session.sessionPreset = .vga640x480

        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: self.currentPosition) else {
            print("Camera not available")
            return
        }

        do {
            let input = try AVCaptureDeviceInput(device: device)
            if session.canAddInput(input) {
                session.addInput(input)
            }
        } catch {
            print("Cannot create camera input: \(error)")
            return
        }

        let output = AVCaptureVideoDataOutput()
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: DispatchQueue(label: "camera.video.queue"))

        if session.canAddOutput(output) {
            session.addOutput(output)
            self.videoOutput = output
        }

        if let connection = output.connection(with: .video) {
            connection.videoOrientation = .portrait
        }
    }
}

extension CameraManager: AVCaptureVideoDataOutputSampleBufferDelegate {
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        delegate?.cameraManager(self, didOutput: pixelBuffer)
    }
}
