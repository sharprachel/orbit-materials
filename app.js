import {
  AutoImageProcessor,
  SegformerForSemanticSegmentation,
  env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

document.addEventListener("DOMContentLoaded", async function () {

  const imageUpload = document.getElementById("image-upload");
  const imagePreview = document.getElementById("image-preview");
  const uploadPlaceholder = document.getElementById("upload-placeholder");
  const clearButton = document.getElementById("clear-button");
  const analyzeButton = document.getElementById("analyze-button");

  // Tell Transformers.js that our model files are stored locally
  env.allowLocalModels = true;

  const MODEL_PATH = "./model";

  let processor = null;
  let model = null;

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

    const reader = new FileReader();

    reader.onload = function (e) {
      imagePreview.src = e.target.result;
      imagePreview.style.display = "block";
      uploadPlaceholder.style.display = "none";
    };

    reader.readAsDataURL(file);
  });

  clearButton.addEventListener("click", function (event) {
    event.preventDefault();

    imageUpload.value = "";
    imagePreview.removeAttribute("src");
    imagePreview.style.display = "none";
    uploadPlaceholder.style.display = "block";
  });

  analyzeButton.textContent = "Loading Model...";

  await loadModel();
});
