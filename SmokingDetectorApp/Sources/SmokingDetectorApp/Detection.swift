import CoreGraphics
import UIKit

enum DetectionClass: Int, CaseIterable {
    case person = 0
    case smoke = 1

    var label: String {
        switch self {
        case .person: return "Person"
        case .smoke: return "Smoke"
        }
    }

    var color: UIColor {
        switch self {
        case .person: return .systemBlue
        case .smoke: return .systemOrange
        }
    }
}

struct Detection {
    let bbox: CGRect
    let confidence: Float
    let cls: DetectionClass

    var isSmokingRelated: Bool {
        cls == .person || cls == .smoke
    }
}
