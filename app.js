import {
  AutoImageProcessor,
  SegformerForSemanticSegmentation,
  RawImage,
  env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

document.addEventListener("DOMContentLoaded", async function () {

  const imageUpload = document.getElementById("image-upload");
  const imagePreview = document.getElementById("image-preview");
  const uploadPlaceholder = document.getElementById("upload-placeholder");
  const clearButton = document.getElementById("clear-button");
  const analyzeButton = document.getElementById("analyze-button");

  env.allowLocalModels = true;

  const MODEL_PATH = "./model";

  let processor = null;
  let model = null;
  let uploadedFile = null;

  async function loadModel() {
    try {
      console.log("Loading Orbit Materials model...");

      processor = await AutoImageProcessor.from_pretrained(MODEL_PATH);

      model = await SegformerForSemanticSegmentation.from_pretrained(
        MODEL_PATH,
        {
          dtype: "fp32"
        }
      );

      console.log("✓ Orbit Materials model loaded successfully");
      analyzeButton.textContent = "Analyze Facade";

    } catch (error) {
      console.error("Model loading error:", error);
      analyzeButton.textContent = "Model Load Error";
    }
  }

  imageUpload.addEventListener("change", function (event) {
    const file = event.target.files[0];

    if (!file) return;

    uploadedFile = file;

    const reader = new FileReader();

    reader.onload = function (e) {
      imagePreview.src = e.target.result;
      imagePreview.style.display = "block";
      uploadPlaceholder.style.display = "none";
    };

    reader.readAsDataURL(file);
  });

  analyzeButton.addEventListener("click", async function () {

    if (!uploadedFile) {
      alert("Please upload a facade image first.");
      return;
    }

    if (!processor || !model) {
      alert("The model is still loading. Please wait a moment.");
      return;
    }

    try {
      analyzeButton.textContent = "Analyzing...";
      analyzeButton.disabled = true;

      const imageURL = URL.createObjectURL(uploadedFile);

      const image = await RawImage.fromURL(imageURL);

      console.log("Image loaded:", image.width, image.height);

      const inputs = await processor(image);

      console.log("Processor output:", inputs);

      const outputs = await model(inputs);

      console.log("Model output:", outputs);
      console.log("Logits shape:", outputs.logits.dims);

      analyzeButton.textContent = "Analysis Complete";

    } catch (error) {
      console.error("Analysis error:", error);
      analyzeButton.textContent = "Analysis Error";

    } finally {
      analyzeButton.disabled = false;
    }
  });

  clearButton.addEventListener("click", function (event) {
    event.preventDefault();

    uploadedFile = null;

    imageUpload.value = "";
    imagePreview.removeAttribute("src");
    imagePreview.style.display = "none";
    uploadPlaceholder.style.display = "block";

    analyzeButton.textContent = "Analyze Facade";
  });

  analyzeButton.textContent = "Loading Model...";

  await loadModel();
});
