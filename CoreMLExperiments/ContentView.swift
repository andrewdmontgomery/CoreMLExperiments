import SwiftUI
import CoreML
import Vision
import PhotosUI

struct ContentView: View {
    @State private var inputImage: UIImage?
    @State private var outputImage: UIImage?
    @State private var selectedImageItem: PhotosPickerItem?

    var body: some View {
        VStack {
            if let inputImage {
                Image(uiImage: inputImage)
                    .resizable()
                    .scaledToFit()
                    .frame(height: 250)
                    .padding()
            } else {
                Text("Select an Image")
                    .foregroundColor(.gray)
            }
            
            if let outputImage {
                Image(uiImage: outputImage)
                    .resizable()
                    .scaledToFit()
                    .frame(height: 250)
                    .padding()
            }
            
            PhotosPicker("Choose Image", selection: $selectedImageItem, matching: .images)
                .onChange(of: selectedImageItem) { _, _ in
                    loadSelectedImage()
                }

            if inputImage != nil {
                VStack {
                    Button("FacePaintV1") {
                        applyFacePaintV1()
                    }
                    Button("FacePaintV2") {
                        applyFacePaintV2()
                    }
                }
                .padding()
                .buttonStyle(.borderedProminent)
            }
        }
        .padding()
    }
    
    /// Loads the selected image from PhotosPicker
    private func loadSelectedImage() {
        guard let selectedImageItem else { return }
        Task {
            if let data = try? await selectedImageItem.loadTransferable(type: Data.self),
               let image = UIImage(data: data) {
                inputImage = image
                outputImage = nil
            }
        }
    }

    private func apply(model: VNCoreMLModel, to inputImage: UIImage) {
        let request = VNCoreMLRequest(model: model) { request, error in
            guard let results = request.results as? [VNPixelBufferObservation],
                  let pixelBuffer = results.first?.pixelBuffer else {
                return
            }

            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let context = CIContext()
            if let cgImage = context.createCGImage(ciImage, from: ciImage.extent) {
                DispatchQueue.main.async {
                    self.outputImage = UIImage(cgImage: cgImage)
                }
            }
        }

        let handler = VNImageRequestHandler(cgImage: inputImage.cgImage!, options: [:])
        DispatchQueue.global(qos: .userInitiated).async {
            try? handler.perform([request])
        }

    }
    
    private func applyFacePaintV1() {
        guard let inputImage = inputImage else {
            return
        }
        
        do {
            let model = try MLModel(contentsOf: FacePaintV2.urlOfModelInThisBundle)
            print(model.modelDescription)
        } catch {
            print("❌ Failed to load model:", error)
        }
        
        do {
            let model = try FacePaintV1(configuration: MLModelConfiguration())
            let visionModel = try VNCoreMLModel(for: model.model)
            apply(model: visionModel, to: inputImage)
        } catch {
            print(error)
        }
    }
    
    private func applyFacePaintV2() {
        guard let inputImage = inputImage,
              let model = try? VNCoreMLModel(for: FacePaintV2().model) else {
            return
        }

        apply(model: model, to: inputImage)
    }
}

#Preview {
    ContentView()
}
