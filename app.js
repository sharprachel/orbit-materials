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
  const overlayCanvas = document.getElementById("segmentation-overlay");
  const uploadBox = document.querySelector(".upload-box");

  env.allowLocalModels = true;

  const MODEL_PATH = "./model";

  let processor = null;
  let model = null;
  let uploadedFile = null;


  // =========================================================
  // MATERIAL COLOURS
  // These match your original Colab version
  // =========================================================

  const CLASS_COLORS = {
    1: [220, 20, 60],      // Brick
    2: [160, 82, 45],      // Exterior Door
    3: [169, 169, 169],    // Fiber Cement
    4: [176, 224, 230],    // Glass
    5: [139, 69, 19],      // Shingles
    6: [70, 130, 180],     // Siding
    7: [128, 128, 128],    // Stone
    8: [255, 215, 0],      // Stucco
    9: [34, 139, 34],      // Vegetation
    10: [135, 206, 235]    // Window
  };


  // =========================================================
  // LOAD MODEL
  // =========================================================

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


  // =========================================================
  // POSITION OVERLAY DIRECTLY OVER DISPLAYED PHOTO
  // =========================================================

  function positionOverlayCanvas() {

    const imageRect = imagePreview.getBoundingClientRect();
    const boxRect = uploadBox.getBoundingClientRect();

    overlayCanvas.style.left =
      `${imageRect.left - boxRect.left}px`;

    overlayCanvas.style.top =
      `${imageRect.top - boxRect.top}px`;

    overlayCanvas.style.width =
      `${imageRect.width}px`;

    overlayCanvas.style.height =
      `${imageRect.height}px`;
  }


  // =========================================================
  // CREATE SEGMENTATION OVERLAY
  // =========================================================

  function drawSegmentationOverlay(logits) {

    const dims = logits.dims;

    console.log("Logits dimensions:", dims);

    // Expected:
    // [batch, classes, height, width]

    const numClasses = dims[1];
    const maskHeight = dims[2];
    const maskWidth = dims[3];

    const logitsData = logits.data;

    // Use the actual displayed image dimensions.
    const imageRect = imagePreview.getBoundingClientRect();

    const displayWidth = Math.max(
      1,
      Math.round(imageRect.width)
    );

    const displayHeight = Math.max(
      1,
      Math.round(imageRect.height)
    );

    overlayCanvas.width = displayWidth;
    overlayCanvas.height = displayHeight;

    positionOverlayCanvas();

    const ctx = overlayCanvas.getContext("2d");

    const imageData = ctx.createImageData(
      displayWidth,
      displayHeight
    );

    const pixels = imageData.data;


    // ---------------------------------------------------------
    // For each visible pixel, find the most likely material.
    //
    // The SegFormer output is smaller than the displayed image,
    // so this maps the model result back over the photograph.
    // ---------------------------------------------------------

    for (let y = 0; y < displayHeight; y++) {

      const maskY = Math.min(
        maskHeight - 1,
        Math.floor(y * maskHeight / displayHeight)
      );

      for (let x = 0; x < displayWidth; x++) {

        const maskX = Math.min(
          maskWidth - 1,
          Math.floor(x * maskWidth / displayWidth)
        );

        let bestClass = 0;
        let bestScore = -Infinity;

        for (let classId = 0; classId < numClasses; classId++) {

          const index =
            classId * maskHeight * maskWidth +
            maskY * maskWidth +
            maskX;

          const score = logitsData[index];

          if (score > bestScore) {
            bestScore = score;
            bestClass = classId;
          }
        }


        // Background remains transparent.
        if (bestClass === 0) {
          continue;
        }

        const color = CLASS_COLORS[bestClass];

        if (!color) {
          continue;
        }

        const pixelIndex =
          (y * displayWidth + x) * 4;

        pixels[pixelIndex] = color[0];
        pixels[pixelIndex + 1] = color[1];
        pixels[pixelIndex + 2] = color[2];

        // ~45% transparency, matching your Colab overlay
        pixels[pixelIndex + 3] = 115;
      }
    }

    ctx.putImageData(imageData, 0, 0);

    overlayCanvas.style.display = "block";
  }


  // =========================================================
  // IMAGE UPLOAD
  // =========================================================

  imageUpload.addEventListener("change", function (event) {

    const file = event.target.files[0];

    if (!file) return;

    uploadedFile = file;

    const reader = new FileReader();

    reader.onload = function (e) {

      imagePreview.src = e.target.result;

      imagePreview.style.display = "block";

      uploadPlaceholder.style.display = "none";

      overlayCanvas.style.display = "none";
    };

    reader.readAsDataURL(file);
  });


  // =========================================================
  // ANALYZE
  // =========================================================

  analyzeButton.addEventListener("click", async function () {

    if (!uploadedFile) {

      alert("Please upload a facade image first.");

      return;
    }

    if (!processor || !model) {

      alert(
        "The model is still loading. Please wait a moment."
      );

      return;
    }


    try {

      analyzeButton.textContent = "Analyzing...";
      analyzeButton.disabled = true;

      overlayCanvas.style.display = "none";


      const imageURL =
        URL.createObjectURL(uploadedFile);

      const image =
        await RawImage.fromURL(imageURL);


      console.log(
        "Image loaded:",
        image.width,
        image.height
      );


      const inputs =
        await processor(image);


      const outputs =
        await model(inputs);


      console.log(
        "Model output:",
        outputs
      );


      drawSegmentationOverlay(
        outputs.logits
      );


      URL.revokeObjectURL(imageURL);


      analyzeButton.textContent =
        "Analysis Complete";


    } catch (error) {

      console.error(
        "Analysis error:",
        error
      );

      analyzeButton.textContent =
        "Analysis Error";


    } finally {

      analyzeButton.disabled = false;
    }
  });


  // =========================================================
  // CLEAR
  // =========================================================

  clearButton.addEventListener("click", function (event) {

    event.preventDefault();

    uploadedFile = null;

    imageUpload.value = "";

    imagePreview.removeAttribute("src");

    imagePreview.style.display = "none";

    uploadPlaceholder.style.display =
      "block";

    overlayCanvas.style.display =
      "none";

    const ctx =
      overlayCanvas.getContext("2d");

    ctx.clearRect(
      0,
      0,
      overlayCanvas.width,
      overlayCanvas.height
    );

    analyzeButton.textContent =
      "Analyze Facade";
  });


  // Keep overlay aligned if browser size changes.
  window.addEventListener("resize", function () {

    if (
      overlayCanvas.style.display === "block"
    ) {
      positionOverlayCanvas();
    }
  });


  // =========================================================
  // INITIALIZE
  // =========================================================

  analyzeButton.textContent =
    "Loading Model...";

  await loadModel();

});
