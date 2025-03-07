import SwiftUI
import CoreML
import Vision
import PhotosUI

struct ContentView: View {
    @State private var originalImage: UIImage?
    @State private var currentImage: UIImage?
    @State private var selectedImageItem: PhotosPickerItem?
    @State private var isProcessing = false
    
    private let modelProvider = ModelProvider()

    var body: some View {
        VStack(spacing: 20) {
            // Permanent container for the image (input or output)
            ZStack {
                Rectangle()
                    .foregroundColor(.gray.opacity(0.2))
                if let image = currentImage {
                    Image(uiImage: image.fixedOrientation())
                        .resizable()
                        .scaledToFit()
                } else {
                    Text("No Image")
                        .foregroundColor(.gray)
                }
                if isProcessing {
                    ProgressView("Apply model...")
                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        .foregroundStyle(.white)
                        .padding()
                        .background(Color.black.opacity(0.3))
                        .cornerRadius(8)
                }
            }
            .frame(width: 300, height: 300)
            .cornerRadius(10)
            .overlay(RoundedRectangle(cornerRadius: 10)
                        .stroke(Color.gray, lineWidth: 1))
            
            // Photo selection control
            PhotosPicker("Choose Photo", selection: $selectedImageItem, matching: .images)
                .onChange(of: selectedImageItem) { _, _ in
                    loadSelectedImage()
                }
            
            // Model buttons and revert button
            
            VStack {
                HStack {
                    Button("Original") {
                        // Revert to the original image (if one has been loaded)
                        currentImage = originalImage
                    }
                    Button("FacePaintV1") {
                        apply(model: .facePaintV1, to: originalImage)
                    }
                    Button("FacePaintV2") {
                        apply(model: .facePaintV2, to: originalImage)

                    }
                }
                HStack {
                    Button("Paprika") {
                        apply(model: .paprika, to: originalImage)

                    }
                    Button("CelebA Distill") {
                        apply(model: .celebADistill, to: originalImage)

                    }
                }
            }
            .padding()
            .buttonStyle(.borderedProminent)
            
            Spacer()
        }
        .padding()
    }
    
    // Loads the selected photo, crops it to a square (aspect fill), and sets both original and current images.
    private func loadSelectedImage() {
        guard let selectedImageItem else { return }
        Task {
            if let data = try? await selectedImageItem.loadTransferable(type: Data.self),
               let image = UIImage(data: data) {
                let fixedImage = image.fixedOrientation() // Ensure correct orientation.
                // Crop the image to a square using an aspect fill approach.
                let squareImage = fixedImage.cropToSquare()?.resized(to: .init(width: 1024, height: 1024)) ?? fixedImage
                originalImage = squareImage
                currentImage = squareImage
            }
        }
    }
    
    // Uses Vision to run the Core ML model and update the current image.
    private func apply(model modelName: ModelProvider.ModelName, to inputImage: UIImage?) {
        guard let inputImage else { return }
        
        do {
            let model = try ModelProvider.visionModel(named: modelName)
            
            let request = VNCoreMLRequest(model: model) { request, error in
                defer {
                    DispatchQueue.main.async {
                        self.isProcessing = false
                    }
                }
                
                guard let results = request.results as? [VNPixelBufferObservation],
                      let pixelBuffer = results.first?.pixelBuffer else { return }
                
                let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
                let context = CIContext()
                if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
                    let resultImage = UIImage(cgImage: cgImage).fixedOrientation()
                    DispatchQueue.main.async {
                        self.currentImage = resultImage
                    }
                }
            }
            guard let cgImage = inputImage.cgImage else { return }
            let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
            DispatchQueue.global(qos: .userInitiated).async {
                try? handler.perform([request])
            }

        } catch {
            fatalError("Failed to load model: \(error)")
        }
    }
}

extension UIImage {
    /// Fixes the orientation of the image by redrawing it.
    func fixedOrientation() -> UIImage {
        if imageOrientation == .up { return self }
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        draw(in: CGRect(origin: .zero, size: size))
        let normalizedImage = UIGraphicsGetImageFromCurrentImageContext()
        UIGraphicsEndImageContext()
        return normalizedImage ?? self
    }
    
    /// Crops the image to a centered square (aspect fill).
    func cropToSquare() -> UIImage? {
        let originalWidth  = size.width
        let originalHeight = size.height
        let squareLength = min(originalWidth, originalHeight)
        let x = (originalWidth - squareLength) / 2.0
        let y = (originalHeight - squareLength) / 2.0
        let cropRect = CGRect(x: x, y: y, width: squareLength, height: squareLength)
        
        guard let cgImage = self.cgImage?.cropping(to: cropRect) else { return nil }
        return UIImage(cgImage: cgImage, scale: scale, orientation: imageOrientation)
    }
    
    /// Resizes the image to the given size.
    func resized(to size: CGSize) -> UIImage? {
        UIGraphicsBeginImageContextWithOptions(size, false, self.scale)
        defer { UIGraphicsEndImageContext() }
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext()
    }
}

#Preview {
    ContentView()
}
