import UIKit
import AVFoundation

class ViewController: UIViewController {
    private let cameraManager = CameraManager()
    private let smokingDetector = SmokingDetector()

    private let videoPreviewView = UIView()
    private let overlayView = BoundingBoxOverlayView()
    private let fpsLabel = UILabel()
    private let infoLabel = UILabel()
    private let cameraSwitchButton = UIButton(type: .system)

    private var lastProcessingTime: CFTimeInterval = 0
    private var frameCount: Int = 0
    private var lastFPSUpdate: CFTimeInterval = 0

    override func viewDidLoad() {
        super.viewDidLoad()
        setupUI()
        cameraManager.delegate = self
        smokingDetector.delegate = self
    }

    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        cameraManager.start()
    }

    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        cameraManager.stop()
    }

    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        let layer = cameraManager.previewLayer
        if layer.superlayer == nil {
            videoPreviewView.layer.insertSublayer(layer, at: 0)
        }
        layer.frame = videoPreviewView.bounds
        overlayView.frame = videoPreviewView.bounds
    }

    private func setupUI() {
        view.backgroundColor = .black

        videoPreviewView.translatesAutoresizingMaskIntoConstraints = false
        overlayView.translatesAutoresizingMaskIntoConstraints = false
        fpsLabel.translatesAutoresizingMaskIntoConstraints = false
        infoLabel.translatesAutoresizingMaskIntoConstraints = false
        cameraSwitchButton.translatesAutoresizingMaskIntoConstraints = false

        view.addSubview(videoPreviewView)
        videoPreviewView.addSubview(overlayView)
        view.addSubview(fpsLabel)
        view.addSubview(infoLabel)
        view.addSubview(cameraSwitchButton)

        NSLayoutConstraint.activate([
            videoPreviewView.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor),
            videoPreviewView.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            videoPreviewView.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            videoPreviewView.bottomAnchor.constraint(equalTo: view.bottomAnchor),

            overlayView.topAnchor.constraint(equalTo: videoPreviewView.topAnchor),
            overlayView.leadingAnchor.constraint(equalTo: videoPreviewView.leadingAnchor),
            overlayView.trailingAnchor.constraint(equalTo: videoPreviewView.trailingAnchor),
            overlayView.bottomAnchor.constraint(equalTo: videoPreviewView.bottomAnchor),

            fpsLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            fpsLabel.trailingAnchor.constraint(equalTo: view.trailingAnchor, constant: -12),

            infoLabel.topAnchor.constraint(equalTo: view.safeAreaLayoutGuide.topAnchor, constant: 12),
            infoLabel.leadingAnchor.constraint(equalTo: view.leadingAnchor, constant: 12),

            cameraSwitchButton.centerXAnchor.constraint(equalTo: view.centerXAnchor),
            cameraSwitchButton.bottomAnchor.constraint(equalTo: view.safeAreaLayoutGuide.bottomAnchor, constant: -24),
            cameraSwitchButton.widthAnchor.constraint(equalToConstant: 64),
            cameraSwitchButton.heightAnchor.constraint(equalToConstant: 64)
        ])

        fpsLabel.textColor = .white
        fpsLabel.font = UIFont.monospacedDigitSystemFont(ofSize: 14, weight: .medium)
        fpsLabel.backgroundColor = UIColor.black.withAlphaComponent(0.5)

        updateInfoLabel()
        infoLabel.numberOfLines = 0
        infoLabel.textColor = .white
        infoLabel.font = UIFont.systemFont(ofSize: 12, weight: .medium)
        infoLabel.backgroundColor = UIColor.black.withAlphaComponent(0.5)

        cameraSwitchButton.backgroundColor = UIColor.white.withAlphaComponent(0.3)
        cameraSwitchButton.layer.cornerRadius = 32
        cameraSwitchButton.setImage(UIImage(systemName: "camera.rotate.fill"), for: .normal)
        cameraSwitchButton.tintColor = .white
        cameraSwitchButton.addTarget(self, action: #selector(didTapSwitchCamera), for: .touchUpInside)
    }

    private func updateInfoLabel() {
        let position = cameraManager.cameraPosition
        let cameraName = (position == .front) ? "Front camera" : "Back camera"
        infoLabel.text = "Smoking Detector\n\(cameraName) | YOLO26n + CoreML"
    }

    @objc private func didTapSwitchCamera() {
        cameraManager.switchCamera()
    }

    private func updateFPS() {
        frameCount += 1
        let now = CACurrentMediaTime()
        if now - lastFPSUpdate >= 1.0 {
            let fps = Double(frameCount) / (now - lastFPSUpdate)
            fpsLabel.text = String(format: "%.1f FPS", fps)
            frameCount = 0
            lastFPSUpdate = now
        }
    }
}

extension ViewController: CameraManagerDelegate {
    func cameraManager(_ manager: CameraManager, didOutput pixelBuffer: CVPixelBuffer) {
        DispatchQueue.main.async { [weak self] in
            self?.updateFPS()
        }

        let now = CACurrentMediaTime()
        guard now - lastProcessingTime >= 0.08 else { return }
        lastProcessingTime = now

        smokingDetector.process(pixelBuffer: pixelBuffer)
    }

    func cameraManager(_ manager: CameraManager, didSwitchTo position: AVCaptureDevice.Position) {
        updateInfoLabel()
    }
}

extension ViewController: SmokingDetectorDelegate {
    func smokingDetector(_ detector: SmokingDetector, didUpdate detections: [Detection], isSmoking: Bool) {
        overlayView.update(detections: detections, isSmoking: isSmoking)
    }
}
