import UIKit

final class BoundingBoxOverlayView: UIView {
    private var detections: [Detection] = []
    private var isSmoking: Bool = false
    private var borderLayer: CAShapeLayer?

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
    }

    required init?(coder: NSCoder) {
        super.init(coder: coder)
        backgroundColor = .clear
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        updateBorderPath()
    }

    func update(detections: [Detection], isSmoking: Bool) {
        self.detections = detections

        if isSmoking != self.isSmoking {
            self.isSmoking = isSmoking
            if isSmoking {
                startBlinking()
            } else {
                stopBlinking()
            }
        }
        setNeedsDisplay()
    }

    private func startBlinking() {
        stopBlinking()

        let layer = CAShapeLayer()
        layer.fillColor = UIColor.clear.cgColor
        layer.strokeColor = UIColor.systemRed.withAlphaComponent(0.9).cgColor
        layer.lineWidth = 6
        layer.shadowColor = UIColor.systemRed.cgColor
        layer.shadowOffset = .zero
        layer.shadowRadius = 16
        layer.shadowOpacity = 1.0
        layer.zPosition = 100

        let borderRect = bounds.insetBy(dx: 3, dy: 3)
        let path = UIBezierPath(
            roundedRect: borderRect,
            byRoundingCorners: [.bottomLeft, .bottomRight],
            cornerRadii: CGSize(width: 47, height: 47)
        )
        layer.path = path.cgPath

        self.layer.addSublayer(layer)
        self.borderLayer = layer

        let animation = CABasicAnimation(keyPath: "opacity")
        animation.fromValue = 1.0
        animation.toValue = 0.2
        animation.duration = 0.7
        animation.autoreverses = true
        animation.repeatCount = .infinity
        animation.timingFunction = CAMediaTimingFunction(name: .easeInEaseOut)
        layer.add(animation, forKey: "pulse")
    }

    private func stopBlinking() {
        borderLayer?.removeFromSuperlayer()
        borderLayer = nil
    }

    private func updateBorderPath() {
        guard let layer = borderLayer else { return }
        let borderRect = bounds.insetBy(dx: 3, dy: 3)
        let path = UIBezierPath(
            roundedRect: borderRect,
            byRoundingCorners: [.bottomLeft, .bottomRight],
            cornerRadii: CGSize(width: 47, height: 47)
        )
        layer.path = path.cgPath
    }

    override func draw(_ rect: CGRect) {
        guard let context = UIGraphicsGetCurrentContext() else { return }

        for det in detections {
            let box = det.bbox
            let pixelRect = CGRect(
                x: box.origin.x * bounds.width,
                y: box.origin.y * bounds.height,
                width: box.width * bounds.width,
                height: box.height * bounds.height
            )

            let color = det.cls.color
            let roundedRect = UIBezierPath(roundedRect: pixelRect, cornerRadius: 8)

            // Glow / shadow effect
            context.setShadow(offset: .zero, blur: 8, color: color.cgColor)
            color.setStroke()
            roundedRect.lineWidth = 2.5
            roundedRect.stroke()
            context.setShadow(offset: .zero, blur: 0, color: nil)

            // Label background (rounded top-left tag)
            let label = String(format: "%@ %.0f%%", det.cls.label, det.confidence * 100)
            let attrs: [NSAttributedString.Key: Any] = [
                .font: UIFont.systemFont(ofSize: 11, weight: .semibold),
                .foregroundColor: UIColor.white
            ]
            let textSize = (label as NSString).size(withAttributes: attrs)
            let tagWidth = textSize.width + 10
            let tagHeight = textSize.height + 6
            let tagRect = CGRect(
                x: pixelRect.minX,
                y: max(pixelRect.minY - tagHeight + 2, 0),
                width: tagWidth,
                height: tagHeight
            )

            let tagPath = UIBezierPath(roundedRect: tagRect, byRoundingCorners: [.topLeft, .topRight, .bottomRight], cornerRadii: CGSize(width: 6, height: 6))
            color.setFill()
            tagPath.fill()

            (label as NSString).draw(at: CGPoint(x: tagRect.minX + 5, y: tagRect.minY + 3), withAttributes: attrs)
        }

        if isSmoking {
            drawSmokingWarning(in: context)
        }
    }

    private func drawSmokingWarning(in context: CGContext) {
        let warningText = "КУРЕНИЕ ОБНАРУЖЕНО"

        let textAttrs: [NSAttributedString.Key: Any] = [
            .font: UIFont.systemFont(ofSize: 26, weight: .black),
            .foregroundColor: UIColor.white
        ]
        let textSize = (warningText as NSString).size(withAttributes: textAttrs)

        let paddingH: CGFloat = 20
        let paddingV: CGFloat = 10
        let bannerWidth = textSize.width + paddingH * 2
        let bannerHeight = textSize.height + paddingV * 2
        let bannerRect = CGRect(
            x: (bounds.width - bannerWidth) / 2,
            y: 52,
            width: bannerWidth,
            height: bannerHeight
        )

        // Red background
        let bgPath = UIBezierPath(roundedRect: bannerRect, cornerRadius: 12)
        UIColor.systemRed.withAlphaComponent(0.92).setFill()
        bgPath.fill()

        // Text
        let textPoint = CGPoint(
            x: bannerRect.midX - textSize.width / 2,
            y: bannerRect.midY - textSize.height / 2
        )
        (warningText as NSString).draw(at: textPoint, withAttributes: textAttrs)
    }
}
