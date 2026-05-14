import CoreML
import Vision
import CoreImage

protocol SmokingDetectorDelegate: AnyObject {
    func smokingDetector(_ detector: SmokingDetector, didUpdate detections: [Detection], isSmoking: Bool)
}

final class SmokingDetector {
    weak var delegate: SmokingDetectorDelegate?

    private var request: VNCoreMLRequest?
    private let confidenceThreshold: Float = 0.25
    private let iouThreshold: Float = 0.5

    init() {
        setupModel()
    }

    private func setupModel() {
        // Try multiple ways to load the model (Xcode compiles .mlpackage to .mlmodelc)
        var model: MLModel? = nil

        // 1. Try compiled .mlmodelc in bundle
        if let compiledURL = Bundle.main.url(forResource: "cigarette_yolo26n", withExtension: "mlmodelc") {
            do {
                let config = MLModelConfiguration()
                config.computeUnits = .all
                model = try MLModel(contentsOf: compiledURL, configuration: config)
                print("Loaded compiled model from: \(compiledURL.lastPathComponent)")
            } catch {
                print("Failed to load compiled model: \(error)")
            }
        }

        // 2. Try raw .mlpackage (for simulator builds that don't compile it)
        if model == nil, let packageURL = Bundle.main.url(forResource: "cigarette_yolo26n", withExtension: "mlpackage") {
            do {
                let config = MLModelConfiguration()
                config.computeUnits = .all
                model = try MLModel(contentsOf: packageURL, configuration: config)
                print("Loaded model from: \(packageURL.lastPathComponent)")
            } catch {
                print("Failed to load mlpackage: \(error)")
            }
        }

        // 3. Try generated Swift class via NSClassFromString
        if model == nil {
            if let generatedClass = NSClassFromString("cigarette_yolo26n") as? NSObject.Type,
               let instance = generatedClass.init() as? MLModel {
                model = instance
                print("Loaded model from generated class")
            }
        }

        guard let loadedModel = model else {
            print("CoreML model not found in bundle (tried .mlmodelc, .mlpackage, generated class)")
            return
        }

        do {
            let vnModel = try VNCoreMLModel(for: loadedModel)
            let request = VNCoreMLRequest(model: vnModel) { [weak self] request, error in
                self?.handleRequest(request, error: error)
            }
            request.imageCropAndScaleOption = .scaleFill
            self.request = request
            print("CoreML request ready")
        } catch {
            print("Failed to create VNCoreMLModel: \(error)")
        }
    }

    func process(pixelBuffer: CVPixelBuffer) {
        guard let request = request else { return }

        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer, orientation: .up, options: [:])
        do {
            try handler.perform([request])
        } catch {
            print("Vision request failed: \(error)")
        }
    }

    private func handleRequest(_ request: VNRequest, error: Error?) {
        if let error = error {
            print("Inference error: \(error)")
            return
        }

        guard let observations = request.results as? [VNCoreMLFeatureValueObservation],
              let first = observations.first,
              let multiArray = first.featureValue.multiArrayValue else {
            print("Unexpected output type")
            return
        }

        let detections = parseDetections(from: multiArray)
        let isSmoking = checkSmoking(detections: detections)

        DispatchQueue.main.async { [weak self] in
            guard let self = self else { return }
            self.delegate?.smokingDetector(self, didUpdate: detections, isSmoking: isSmoking)
        }
    }

    private func parseDetections(from multiArray: MLMultiArray) -> [Detection] {
        let shape = multiArray.shape.map { $0.intValue }
        guard shape.count == 3, shape[0] == 1, shape[2] == 6 else {
            print("Unexpected multiArray shape: \(shape)")
            return []
        }

        let numDetections = shape[1]
        var detections: [Detection] = []

        for i in 0..<numDetections {
            let x1 = floatValue(multiArray, i, 0)
            let y1 = floatValue(multiArray, i, 1)
            let x2 = floatValue(multiArray, i, 2)
            let y2 = floatValue(multiArray, i, 3)
            let conf = floatValue(multiArray, i, 4)
            let cls = Int(floatValue(multiArray, i, 5))

            guard conf >= confidenceThreshold else { continue }
            guard let detectionClass = DetectionClass(rawValue: cls) else { continue }

            let isNormalized = max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5
            let w: CGFloat = isNormalized ? 1.0 : 640.0
            let h: CGFloat = isNormalized ? 1.0 : 640.0

            let rect = CGRect(
                x: CGFloat(x1) / w,
                y: CGFloat(y1) / h,
                width: (CGFloat(x2) - CGFloat(x1)) / w,
                height: (CGFloat(y2) - CGFloat(y1)) / h
            )

            detections.append(Detection(bbox: rect, confidence: conf, cls: detectionClass))
        }

        return nonMaxSuppression(detections: detections)
    }

    private func floatValue(_ multiArray: MLMultiArray, _ row: Int, _ col: Int) -> Float {
        let index = [0, row, col] as [NSNumber]
        return multiArray[index].floatValue
    }

    private func nonMaxSuppression(detections: [Detection]) -> [Detection] {
        let sorted = detections.sorted { $0.confidence > $1.confidence }
        var result: [Detection] = []

        for det in sorted {
            var shouldSuppress = false
            for kept in result where kept.cls == det.cls {
                if iou(det.bbox, kept.bbox) > iouThreshold {
                    shouldSuppress = true
                    break
                }
            }
            if !shouldSuppress {
                result.append(det)
            }
        }

        return result
    }

    private func iou(_ a: CGRect, _ b: CGRect) -> Float {
        let intersection = a.intersection(b)
        guard !intersection.isNull else { return 0 }
        let interArea = intersection.width * intersection.height
        let unionArea = a.width * a.height + b.width * b.height - interArea
        guard unionArea > 0 else { return 0 }
        return Float(interArea / unionArea)
    }

    private func checkSmoking(detections: [Detection]) -> Bool {
        let persons = detections.filter { $0.cls == .person }
        let smokes = detections.filter { $0.cls == .smoke }

        for person in persons {
            for smoke in smokes {
                if person.bbox.intersects(smoke.bbox) {
                    return true
                }
            }
        }
        return false
    }
}
