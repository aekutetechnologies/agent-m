// Read-only OCR utility: prints the recognized text of an image using the
// macOS Vision framework. Usage: swift ocr.swift <image-path>
import Vision
import AppKit

guard CommandLine.arguments.count > 1 else {
    print("usage: ocr <image>")
    exit(1)
}
let path = CommandLine.arguments[1]
guard let img = NSImage(contentsOfFile: path),
      let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) else {
    print("ERR: cannot load image \(path)")
    exit(1)
}
let request = VNRecognizeTextRequest { request, error in
    guard let observations = request.results as? [VNRecognizedTextObservation] else { return }
    // Sort top-to-bottom so the layout reads naturally.
    let sorted = observations.sorted { $0.boundingBox.midY > $1.boundingBox.midY }
    for observation in sorted {
        if let candidate = observation.topCandidates(1).first {
            print(candidate.string)
        }
    }
}
request.recognitionLevel = .accurate
request.usesLanguageCorrection = true
let handler = VNImageRequestHandler(cgImage: cg, options: [:])
do {
    try handler.perform([request])
} catch {
    print("ERR: \(error)")
    exit(1)
}
