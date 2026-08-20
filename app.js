import {
  AutoImageProcessor,
  SegformerForSemanticSegmentation,
  RawImage,
  env
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";


document.addEventListener("DOMContentLoaded", async function () {

  // =========================================================
  // PAGE ELEMENTS
  // =========================================================

  const imageUpload =
    document.getElementById("image-upload");

  const imagePreview =
    document.getElementById("image-preview");

  const uploadPlaceholder =
    document.getElementById("upload-placeholder");

  const clearButton =
    document.getElementById("clear-button");

  const analyzeButton =
    document.getElementById("analyze-button");

  const overlayCanvas =
    document.getElementById("segmentation-overlay");

  const uploadBox =
    document.querySelector(".upload-box");

  const componentsResults =
    document.getElementById("components-results");

  const recommendationsResults =
    document.getElementById("recommendations-results");


  // =========================================================
  // MODEL
  // =========================================================

  env.allowLocalModels = true;

  const MODEL_PATH = "./model";

  let processor = null;
  let model = null;
  let uploadedFile = null;


  // =========================================================
  // MODEL CLASSES
  // =========================================================

  const CLASS_COLORS = {
    1: [220, 20, 60],
    2: [160, 82, 45],
    3: [169, 169, 169],
    4: [176, 224, 230],
    5: [139, 69, 19],
    6: [70, 130, 180],
    7: [128, 128, 128],
    8: [255, 215, 0],
    9: [34, 139, 34],
    10: [135, 206, 235]
  };


  const CLASS_NAMES = {
    0: "Background",
    1: "Brick",
    2: "Door & Frame",
    3: "Fiber Cement",
    4: "Non-window Glass",
    5: "Shingles",
    6: "Siding",
    7: "Stone",
    8: "Stucco",
    9: "Vegetation",
    10: "Window & Frame"
  };


  // =========================================================
  // COMPONENT DISPLAY
  // =========================================================

  const COMPONENT_INFO = {

    1: {
      name: "Brick",
      svg: "./assets/svg/Brick.svg"
    },

    2: {
      name: "Door & Frame",
      svg: "./assets/svg/Door.svg"
    },

    3: {
      name: "Fiber Cement",
      svg: "./assets/svg/Siding.svg"
    },

    5: {
      name: "Shingles",
      svg: "./assets/svg/Shingles.svg"
    },

    6: {
      name: "Siding",
      svg: "./assets/svg/Siding.svg"
    },

    7: {
      name: "Stone",
      svg: "./assets/svg/Stone.svg"
    },

    8: {
      name: "Stucco",
      svg: "./assets/svg/Stucco.svg"
    },

    10: {
      name: "Window & Frame",
      svg: "./assets/svg/Window.svg"
    }
  };


  // Classes excluded from the user-facing component display.
  const HIDDEN_VISIBLE_CLASSES =
    new Set([
      0, // background
      4, // non-window glass
      9  // vegetation
    ]);


  // =========================================================
  // HIDDEN COMPONENT INFERENCE
  // =========================================================

  // Preserves the logic from the original Colab.
  // These visible facade materials suggest wood framing behind.
  const WOOD_FRAMING_TRIGGERS =
    new Set([
      1, // Brick
      3, // Fiber Cement
      5, // Shingles
      6, // Siding
      8  // Stucco
    ]);


  // =========================================================
  // CIRCULARITY INFORMATION
  // =========================================================

  const RECOMMENDATION_INFO = {

    1: {
      name: "Brick",
      potential: "High",
      text:
        "Prioritize direct salvage and reuse. Brick can be cleaned and reused in new construction when removed intact.",
      anchor: "brick-stone-masonry"
    },

    2: {
      name: "Door & Frame",
      potential: "Very High",
      text:
        "Remove the door, frame and hardware intact where possible. Quality doors have excellent potential for direct architectural reuse.",
      anchor: "door-frame"
    },

    3: {
      name: "Fiber Cement",
      potential: "Limited",
      text:
        "Direct reuse is often limited by brittleness and removal damage. Careful separation can improve recycling and waste-diversion opportunities.",
      anchor: "building-envelope-panel"
    },

    5: {
      name: "Shingles",
      potential: "Limited",
      text:
        "Removal often damages shingles, so recovery may focus more on recycling or waste diversion than direct reuse.",
      anchor: "building-envelope-panel"
    },

    6: {
      name: "Siding",
      potential: "Medium",
      text:
        "Carefully removed panels may be suitable for reuse depending on material, fastening method and condition.",
      anchor: "building-envelope-panel"
    },

    7: {
      name: "Stone",
      potential: "Very High",
      text:
        "Prioritize whole-piece salvage. Natural stone has excellent reuse value and avoids the impacts associated with new quarrying.",
      anchor: "brick-stone-masonry"
    },

    8: {
      name: "Stucco",
      potential: "Limited",
      text:
        "Stucco is normally bonded to its substrate and is rarely reused intact. Recovery generally focuses on separation and waste reduction.",
      anchor: "building-envelope-panel"
    },

    10: {
      name: "Window & Frame",
      potential: "High",
      text:
        "Remove the complete assembly where possible. Intact windows and frames may be reused directly or separated for component recovery.",
      anchor: "window-frame"
    }
  };


  // =========================================================
  // LOAD MODEL
  // =========================================================

  async function loadModel() {

    try {

      console.log(
        "Loading Orbit Materials model..."
      );


      processor =
        await AutoImageProcessor.from_pretrained(
          MODEL_PATH
        );


      model =
        await SegformerForSemanticSegmentation.from_pretrained(
          MODEL_PATH,
          {
            dtype: "fp32"
          }
        );


      console.log(
        "✓ Orbit Materials model loaded"
      );


      analyzeButton.textContent =
        "Analyze Facade";


    } catch (error) {

      console.error(
        "Model loading error:",
        error
      );


      analyzeButton.textContent =
        "Model Load Error";
    }
  }


  // =========================================================
  // POSITION OVERLAY
  // =========================================================

  function positionOverlayCanvas() {

    const imageRect =
      imagePreview.getBoundingClientRect();

    const boxRect =
      uploadBox.getBoundingClientRect();


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
  // FIND DETECTED MATERIALS
  // =========================================================

  function getDetectedMaterials(logits) {

    const dims = logits.dims;

    const numClasses = dims[1];
    const maskHeight = dims[2];
    const maskWidth = dims[3];

    const logitsData = logits.data;

    const counts =
      new Array(numClasses).fill(0);

    const totalPixels =
      maskHeight * maskWidth;


    for (
      let y = 0;
      y < maskHeight;
      y++
    ) {

      for (
        let x = 0;
        x < maskWidth;
        x++
      ) {

        let bestClass = 0;
        let bestScore = -Infinity;


        for (
          let classId = 0;
          classId < numClasses;
          classId++
        ) {

          const index =
            classId * maskHeight * maskWidth +
            y * maskWidth +
            x;


          const score =
            logitsData[index];


          if (score > bestScore) {

            bestScore = score;
            bestClass = classId;
          }
        }


        counts[bestClass]++;
      }
    }


    const detected = [];


    for (
      let classId = 0;
      classId < numClasses;
      classId++
    ) {

      if (counts[classId] === 0) {
        continue;
      }


      detected.push({

        classId: classId,

        name:
          CLASS_NAMES[classId],

        count:
          counts[classId],

        percent:
          (
            counts[classId] /
            totalPixels
          ) * 100
      });
    }


    detected.sort(
      (a, b) =>
        b.percent - a.percent
    );


    return detected;
  }


  // =========================================================
  // DETECTED COMPONENTS
  // =========================================================

 function renderComponentCards(detectedMaterials) {

  const visibleComponents =
    detectedMaterials.filter(item =>
      !HIDDEN_VISIBLE_CLASSES.has(item.classId) &&
      COMPONENT_INFO[item.classId]
    );


  if (visibleComponents.length === 0) {

    componentsResults.innerHTML = `
      <p class="muted">
        No reusable components detected.
      </p>
    `;

    return;
  }


  const anchorMap = {
    1: "brick-stone-masonry",
    2: "door-frame",
    3: "building-envelope-panel",
    5: "building-envelope-panel",
    6: "building-envelope-panel",
    7: "brick-stone-masonry",
    8: "building-envelope-panel",
    10: "window-frame"
  };


  let html = `
    <div class="component-grid">
  `;


  visibleComponents.forEach(item => {

    const component =
      COMPONENT_INFO[item.classId];

    const color =
      CLASS_COLORS[item.classId];

    const anchor =
      anchorMap[item.classId] ||
      "material-info";


    html += `
      <a
        class="component-card component-card-link"
        href="#${anchor}"
      >

        <div class="component-image-wrap">

          <img
            src="${component.svg}"
            alt="${component.name}"
            class="component-image"
          >

        </div>

        <div class="component-label">

          <span
            class="component-swatch"
            style="
              background:
              rgb(
                ${color[0]},
                ${color[1]},
                ${color[2]}
              );
            "
          ></span>

          <span>
            ${component.name}
          </span>

        </div>

      </a>
    `;
  });


  html += `
    </div>
  `;


  const hasWoodFraming =
    detectedMaterials.some(
      item =>
        WOOD_FRAMING_TRIGGERS.has(
          item.classId
        )
    );


  if (hasWoodFraming) {

    html += `
      <div class="hidden-components">

        <div class="hidden-components-title">
          Likely Hidden Components
        </div>

        <div class="hidden-components-copy">
          Based on the detected cladding system,
          the facade likely includes structural
          components behind the visible exterior.
        </div>

        <a
          class="hidden-component-card"
          href="#dimensional-lumber"
        >

          <img
            src="./assets/svg/Wood Frame.svg"
            alt="Dimensional Lumber"
            class="hidden-component-image"
          >

          <div>
            Dimensional Lumber
          </div>

        </a>

      </div>
    `;
  }


  componentsResults.innerHTML =
    html;
}

  // =========================================================
  // CIRCULARITY RECOMMENDATIONS
  // =========================================================

  function renderRecommendations(
    detectedMaterials
  ) {

    const materials =
      detectedMaterials.filter(
        item =>
          RECOMMENDATION_INFO[
            item.classId
          ]
      );


    if (materials.length === 0) {

      recommendationsResults.innerHTML = `
        <p class="muted">
          No reuse recommendations available
          for the detected materials.
        </p>
      `;

      return;
    }


    let html = `
      <div class="recommendations-list">
    `;


    materials.forEach(item => {

      const info =
        RECOMMENDATION_INFO[
          item.classId
        ];


      html += `
        <div class="recommendation-card">

          <div class="recommendation-header">

            <span class="recommendation-name">
              ${info.name}
            </span>

            <span class="recommendation-badge">
              ${info.potential}
            </span>

          </div>

          <div class="recommendation-copy">
            ${info.text}
          </div>

          <a
            href="#${info.anchor}"
            class="recommendation-link"
          >
            View Material Information →
          </a>

        </div>
      `;
    });


    html += `
      </div>
    `;


    recommendationsResults.innerHTML =
      html;
  }


  // =========================================================
  // SEGMENTATION OVERLAY
  // =========================================================

  function drawSegmentationOverlay(
    logits
  ) {

    const dims =
      logits.dims;

    const numClasses =
      dims[1];

    const maskHeight =
      dims[2];

    const maskWidth =
      dims[3];

    const logitsData =
      logits.data;


    const imageRect =
      imagePreview.getBoundingClientRect();


    const displayWidth =
      Math.max(
        1,
        Math.round(
          imageRect.width
        )
      );


    const displayHeight =
      Math.max(
        1,
        Math.round(
          imageRect.height
        )
      );


    overlayCanvas.width =
      displayWidth;

    overlayCanvas.height =
      displayHeight;


    positionOverlayCanvas();


    const ctx =
      overlayCanvas.getContext(
        "2d"
      );


    const imageData =
      ctx.createImageData(
        displayWidth,
        displayHeight
      );


    const pixels =
      imageData.data;


    for (
      let y = 0;
      y < displayHeight;
      y++
    ) {

      const maskY =
        Math.min(
          maskHeight - 1,

          Math.floor(
            y *
            maskHeight /
            displayHeight
          )
        );


      for (
        let x = 0;
        x < displayWidth;
        x++
      ) {

        const maskX =
          Math.min(
            maskWidth - 1,

            Math.floor(
              x *
              maskWidth /
              displayWidth
            )
          );


        let bestClass = 0;
        let bestScore =
          -Infinity;


        for (
          let classId = 0;
          classId < numClasses;
          classId++
        ) {

          const index =
            classId *
            maskHeight *
            maskWidth +

            maskY *
            maskWidth +

            maskX;


          const score =
            logitsData[index];


          if (score > bestScore) {

            bestScore =
              score;

            bestClass =
              classId;
          }
        }


        if (
          bestClass === 0
        ) {
          continue;
        }


        const color =
          CLASS_COLORS[
            bestClass
          ];


        if (!color) {
          continue;
        }


        const pixelIndex =
          (
            y *
            displayWidth +
            x
          ) * 4;


        pixels[pixelIndex] =
          color[0];

        pixels[pixelIndex + 1] =
          color[1];

        pixels[pixelIndex + 2] =
          color[2];

        pixels[pixelIndex + 3] =
          115;
      }
    }


    ctx.putImageData(
      imageData,
      0,
      0
    );


    overlayCanvas.style.display =
      "block";
  }


  // =========================================================
  // RESET RESULTS
  // =========================================================

  function resetResults() {

    componentsResults.innerHTML = `
      <p class="muted">
        Detected components will appear
        here after analysis.
      </p>
    `;


    recommendationsResults.innerHTML = `
      <p class="muted">
        Reuse guidance will appear
        here after analysis.
      </p>
    `;
  }


  // =========================================================
  // IMAGE UPLOAD
  // =========================================================

  imageUpload.addEventListener(
    "change",
    function (event) {

      const file =
        event.target.files[0];


      if (!file) {
        return;
      }


      uploadedFile =
        file;


      const reader =
        new FileReader();


      reader.onload =
        function (e) {

          imagePreview.src =
            e.target.result;


          imagePreview.style.display =
            "block";


          uploadPlaceholder.style.display =
            "none";


          overlayCanvas.style.display =
            "none";


          analyzeButton.textContent =
            "Analyze Facade";


          resetResults();
        };


      reader.readAsDataURL(
        file
      );
    }
  );


  // =========================================================
  // ANALYZE
  // =========================================================

  analyzeButton.addEventListener(
    "click",
    async function () {

      if (!uploadedFile) {

        alert(
          "Please upload a facade image first."
        );

        return;
      }


      if (
        !processor ||
        !model
      ) {

        alert(
          "The model is still loading. Please wait a moment."
        );

        return;
      }


      let imageURL =
        null;


      try {

        analyzeButton.textContent =
          "Analyzing...";


        analyzeButton.disabled =
          true;


        overlayCanvas.style.display =
          "none";


        imageURL =
          URL.createObjectURL(
            uploadedFile
          );


        const image =
          await RawImage.fromURL(
            imageURL
          );


        const inputs =
          await processor(
            image
          );


        const outputs =
          await model(
            inputs
          );


        drawSegmentationOverlay(
          outputs.logits
        );


        const detectedMaterials =
          getDetectedMaterials(
            outputs.logits
          );


        console.log(
          "Detected materials:",
          detectedMaterials
        );


        renderComponentCards(
          detectedMaterials
        );


        renderRecommendations(
          detectedMaterials
        );


        analyzeButton.textContent =
          "Analyze Facade";


      } catch (error) {

        console.error(
          "Analysis error:",
          error
        );


        analyzeButton.textContent =
          "Analysis Error";


      } finally {

        analyzeButton.disabled =
          false;


        if (imageURL) {

          URL.revokeObjectURL(
            imageURL
          );
        }
      }
    }
  );


  // =========================================================
  // CLEAR
  // =========================================================

  clearButton.addEventListener(
    "click",
    function (event) {

      event.preventDefault();


      uploadedFile =
        null;


      imageUpload.value =
        "";


      imagePreview.removeAttribute(
        "src"
      );


      imagePreview.style.display =
        "none";


      uploadPlaceholder.style.display =
        "block";


      overlayCanvas.style.display =
        "none";


      const ctx =
        overlayCanvas.getContext(
          "2d"
        );


      ctx.clearRect(
        0,
        0,
        overlayCanvas.width,
        overlayCanvas.height
      );


      analyzeButton.textContent =
        "Analyze Facade";


      resetResults();
    }
  );


  // =========================================================
  // RESIZE
  // =========================================================

  window.addEventListener(
    "resize",
    function () {

      if (
        overlayCanvas.style.display ===
        "block"
      ) {

        positionOverlayCanvas();
      }
    }
  );


  // =========================================================
  // INITIALIZE
  // =========================================================

  analyzeButton.textContent =
    "Loading Model...";


  resetResults();


  await loadModel();

});
