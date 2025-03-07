import Foundation
import CoreML
import Vision

struct ModelProvider {
    enum ModelName {
        case facePaintV1
        case facePaintV2
        case paprika
        case celebADistill
    }
    
    static func visionModel(named modelName: ModelName) throws -> VNCoreMLModel {
        let config = MLModelConfiguration()
        
        switch modelName {
        case .facePaintV1:
            let model = try FacePaintV1(configuration: config)
            return try VNCoreMLModel(for: model.model)
        case .facePaintV2:
            let model = try FacePaintV2(configuration: config)
            return try VNCoreMLModel(for: model.model)
        case .paprika:
            let model = try Paprika(configuration: config)
            return try VNCoreMLModel(for: model.model)
        case .celebADistill:
            let model = try CelebA_Distill(configuration: config)
            return try VNCoreMLModel(for: model.model)
        }
    }
}
