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
  
  const detectedMaterialInfoSection =
    document.getElementById("detected-material-info");

  const detectedMaterialInfoResults =
    document.getElementById("detected-material-info-results");

  const sampleFacadeButtons =
    document.querySelectorAll(".sample-facade-btn");

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
  1: [230, 35, 70],      // Brick — crimson red
  2: [255, 145, 25],     // Door & Frame — bright orange
  3: [255, 205, 40],     // Fiber Cement — golden yellow
  4: [40, 210, 200],     // Non-window Glass — turquoise
  5: [150, 90, 220],     // Shingles — violet
  6: [45, 105, 220],     // Siding — strong blue
  7: [215, 90, 210],     // Stone — bright magenta
  8: [120, 210, 70],     // Stucco — lime green
  9: [20, 150, 75],      // Vegetation — green
  10: [80, 190, 235]     // Window & Frame — sky blue
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
// MATERIAL INFORMATION SHOWN AFTER ANALYSIS
// =========================================================

const MATERIAL_DETAIL_INFO = {

  "dimensional-lumber": {
    title: "Dimensional Lumber",

    images: [
      {
        src: "./assets/material_info/wood_material_info.svg",
        alt: "Dimensional Lumber"
      }
    ],

    ease:
      "Deconstruction of dimensional lumber is of medium difficulty. Nailed connections must be pried apart, and can risk splitting the wood during removal. Denailing is a labor-intensive activity, and can be time consuming.",

    process:
      "Wood must be denailed either by a manual or pneumatic nail punch. Ends of wood may need trimming if the wood has split. Lumber should be dried and planed if the surface is damaged or to standardize thickness. Any paint or chemicals should be planed or sanded off before reuse."
  },


  "brick-stone-masonry": {
    title: "Brick, Stone and Masonry Units",

    images: [
      {
        src: "./assets/material_info/brick_material_info.svg",
        alt: "Brick"
      },
      {
        src: "./assets/material_info/stone_material_info.svg",
        alt: "Stone"
      }
    ],

    ease:
      "Brick has a high effort for proper deconstruction. Bricks should be removed by hand after mortar is softened, for example with chipping hammers. It is easy to damage bricks if mortar is very hard, which is common in post-1930s cement mortar. Older lime mortar is softer, and bricks often come out intact. Stone blocks can be pried out with equipment, though there is a risk of cracking if not careful.",

    process:
      "Clean mortar traditionally with hammers and chisels. Bricks are then sorted by quality, for example face bricks versus interior bricks. Stone mortar is cleaned off and any embedded metal removed. Stone may be reshaped, resized, washed, or chemically cleaned where required."
  },


  "window-frame": {
    title: "Window & Frame",

    images: [
      {
        src: "./assets/material_info/window_material_info.svg",
        alt: "Window and Frame"
      }
    ],

    ease:
      "Removal can be easy if the frame is screwed in. To remove, unscrew and carefully pry the window out. Old wood windows can be removed intact if weights and stops are removed. There is a high risk of glass breakage if pried roughly or if there are hidden fasteners.",

    process:
      "Wood windows can be stripped, repaired, reglazed, weatherstripped, and repainted. Steel windows may require rust removal and refinishing, while aluminum windows can often be cleaned and have worn gaskets replaced."
  },


  "door-frame": {
    title: "Door & Frame",

    images: [
      {
        src: "./assets/material_info/door_material_info.svg",
        alt: "Door and Frame"
      }
    ],

    ease:
      "Wood doors are generally very easy to remove. Hinge pins can be removed or hinges unscrewed. Metal frames can be more difficult and may bend during removal. Wood frames can often be removed intact once one side of the wall is opened.",

    process:
      "After removal, surfaces can be stripped, sanded, repainted or stained. Frames may require repair or resizing and hardware can be cleaned, repaired, re-keyed or replaced."
  },


  "building-envelope-panel": {
    title: "Building Envelope Panel",

    images: [
      {
        src: "./assets/material_info/siding_material_info.svg",
        alt: "Building Envelope Panel"
      }
    ],

    ease:
      "Panelized systems can be moderately difficult to disassemble depending on the type. Long metal siding panels often come off by removing screws. Wood siding can be pried off but may crack. Stucco and fiber cement generally have more limited reuse potential.",

    process:
      "Remove sealants and gaskets from cladding panels. Check metal panels for corrosion. Wood siding should be denailed, stripped and refinished. Stucco and fiber cement generally require careful separation and waste handling rather than direct salvage."
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
// DETECTED MATERIAL INFORMATION
// =========================================================

function renderDetectedMaterialInfo(detectedMaterials) {

  if (
    !detectedMaterialInfoSection ||
    !detectedMaterialInfoResults
  ) {
    return;
  }


  const anchorOrder = [];
  const seenAnchors = new Set();


  // Add information for directly detected materials

  detectedMaterials.forEach(item => {

    const info =
      RECOMMENDATION_INFO[item.classId];


    if (!info) {
      return;
    }


    const anchor =
      info.anchor;


    if (
      MATERIAL_DETAIL_INFO[anchor] &&
      !seenAnchors.has(anchor)
    ) {

      seenAnchors.add(anchor);

      anchorOrder.push(anchor);
    }

  });


  // Add dimensional lumber when the facade
  // suggests hidden wood framing

  const hasWoodFraming =
    detectedMaterials.some(
      item =>
        WOOD_FRAMING_TRIGGERS.has(
          item.classId
        )
    );


  if (
    hasWoodFraming &&
    !seenAnchors.has("dimensional-lumber")
  ) {

    seenAnchors.add(
      "dimensional-lumber"
    );

    anchorOrder.push(
      "dimensional-lumber"
    );
  }


  // If nothing relevant was detected,
  // keep the section hidden

  if (anchorOrder.length === 0) {

    detectedMaterialInfoSection.hidden =
      true;

    detectedMaterialInfoResults.innerHTML =
      "";

    return;
  }


  let html = "";


  anchorOrder.forEach(anchor => {

    const detail =
      MATERIAL_DETAIL_INFO[anchor];


    const mediaClass =
      detail.images.length > 1
        ? "material-info-media material-duo-grid"
        : "material-info-media";


    const imagesHTML =
      detail.images
        .map(image => `
          <img
            src="${image.src}"
            alt="${image.alt}"
            class="material-diagram-img"
          >
        `)
        .join("");


    html += `

      <div
        id="${anchor}"
        class="material-info-grid"
      >

        <div class="${mediaClass}">
          ${imagesHTML}
        </div>


        <div class="material-info-copy">

          <div class="material-copy-title">
            ${detail.title}
          </div>


          <div class="material-copy-subtitle">
            Ease of Deconstruction
          </div>

          <div class="material-copy-body">
            ${detail.ease}
          </div>


          <div class="material-copy-subtitle">
            Deconstruction Process
          </div>

          <div class="material-copy-body">
            ${detail.process}
          </div>

        </div>

      </div>

    `;

  });


  detectedMaterialInfoResults.innerHTML =
    html;


  detectedMaterialInfoSection.hidden =
    false;
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

  if (
  detectedMaterialInfoSection &&
  detectedMaterialInfoResults
) {

  detectedMaterialInfoSection.hidden =
    true;

  detectedMaterialInfoResults.innerHTML =
    "";
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
// SAMPLE FACADE SELECTION
// =========================================================

sampleFacadeButtons.forEach(button => {

  button.addEventListener(
    "click",
    async function () {

      const samplePath =
        button.dataset.sample;

      if (!samplePath) {
        return;
      }


      try {

        const response =
          await fetch(samplePath);

        const blob =
          await response.blob();


        uploadedFile =
          new File(
            [blob],
            samplePath.split("/").pop(),
            {
              type: blob.type || "image/jpeg"
            }
          );


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
          uploadedFile
        );


      } catch (error) {

        console.error(
          "Could not load sample facade:",
          error
        );

      }

    }
  );

});
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

        renderDetectedMaterialInfo(
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
