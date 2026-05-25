const views = document.querySelectorAll(".view");
const navItems = document.querySelectorAll(".nav-item");
const fileInput = document.querySelector("#fileInput");
const cameraInput = document.querySelector("#cameraInput");
const fileName = document.querySelector("#fileName");
const uploadZone = document.querySelector("#uploadZone");
const uploadPrompt = document.querySelector("#uploadPrompt");
const sourceChoice = document.querySelector("#sourceChoice");
const photoReview = document.querySelector("#photoReview");
const selectedPreview = document.querySelector("#selectedPreview");
const startAnalysisBtn = document.querySelector("#startAnalysisBtn");
const cameraReview = document.querySelector("#cameraReview");
const cameraVideo = document.querySelector("#cameraVideo");
const cameraCanvas = document.querySelector("#cameraCanvas");
const captureBtn = document.querySelector("#captureBtn");
const cameraBackBtn = document.querySelector("#cameraBackBtn");
const focusRing = document.querySelector(".focus-ring");
const apiBaseUrl =
  window.SKIN_API_BASE_URL ||
  window.localStorage.getItem("skinApiBaseUrl") ||
  (window.location.protocol.startsWith("http") ? window.location.origin : null) ||
  "http://127.0.0.1:8000";

let selectedFile = null;
let latestApiResult = null;
let currentDetailItem = null;
let selectedPreviewUrl = null;
let cameraStream = null;

const ANALYSIS_PAGE_COPY = {
  pageTitle: "Visual Skin Analysis",
  pageSubtitle:
    "This page summarizes AI-predicted visual skin concern patterns from the uploaded facial image. The results are intended for educational demonstration and visual interpretation only.",
};

const CONCERN_GUIDANCE = {
  Acne: {
    analysis:
      "This selected area shows visible acne-like patterns such as localized redness, small bumps, or uneven texture.",
    suggestion:
      "Keep the routine simple and barrier-friendly. Gentle cleansing, non-comedogenic moisturizer, and daily sunscreen are reasonable first steps.",
    use: "Gentle cleanser, lightweight moisturizer, daily sunscreen.",
    avoid: "Picking, harsh scrubs, and layering multiple strong actives at once.",
  },
  Blackheads: {
    analysis:
      "This selected area looks most consistent with clogged-pore or open-comedone visual patterns.",
    suggestion:
      "A consistent pore-care routine may help the appearance over time. Introduce exfoliating products slowly and watch for irritation.",
    use: "Salicylic acid used gradually, non-comedogenic moisturizer, sunscreen.",
    avoid: "Aggressive squeezing, frequent pore strips, and abrasive exfoliation.",
  },
  Whiteheads: {
    analysis:
      "This selected area shows closed-comedone-like visual patterns, often appearing as small raised or light bumps.",
    suggestion:
      "Focus on consistency rather than intensity. Gentle exfoliation and a light moisturizer can support a smoother-looking surface.",
    use: "Gentle cleanser, light moisturizer, slow introduction of mild exfoliation.",
    avoid: "Over-cleansing, heavy pore-clogging products, and picking.",
  },
  Papules: {
    analysis:
      "This selected area shows inflamed bump-like visual patterns without a clearly visible white center.",
    suggestion:
      "Minimize friction and avoid picking. A calm, simple routine is best when visible redness or irritation is present.",
    use: "Gentle cleanser, soothing moisturizer, sunscreen.",
    avoid: "Scrubbing, squeezing, and applying many active products together.",
  },
  Pustules: {
    analysis:
      "This selected area shows an inflamed spot-like pattern with a brighter central area.",
    suggestion:
      "Avoid squeezing the area. Keep the skin clean and consider a simple spot-care approach if your skin tolerates it.",
    use: "Gentle cleansing, hydrocolloid patch, lightweight moisturizer.",
    avoid: "Popping, harsh acids on broken skin, and repeated touching.",
  },
  Cyst: {
    analysis:
      "This selected area appears closer to a deeper bump-like visual pattern than a surface-level clogged pore.",
    suggestion:
      "Be gentle with this area and avoid pressure or picking. If it feels painful, persistent, or changes quickly, professional review is a good idea.",
    use: "Gentle routine, minimal friction, sunscreen.",
    avoid: "Squeezing, strong exfoliation, and repeated pressure.",
  },
  Milia: {
    analysis:
      "This selected area shows small firm bump-like visual patterns that can look different from inflamed acne spots.",
    suggestion:
      "Use a lightweight routine and avoid heavy occlusive products around the area. Changes usually take consistency and time.",
    use: "Light moisturizer, sunscreen, gentle cleansing.",
    avoid: "Picking, heavy creams over the area, and aggressive extraction.",
  },
  Eczema: {
    analysis:
      "This selected area shows dryness, redness, or uneven tone patterns that look closer to irritation-prone skin texture.",
    suggestion:
      "Prioritize barrier support. Keep the routine gentle, hydrating, and fragrance-free where possible.",
    use: "Fragrance-free moisturizer, gentle cleanser, sunscreen.",
    avoid: "Harsh exfoliation, fragrance-heavy products, and over-cleansing.",
  },
  Rosacea: {
    analysis:
      "This selected area shows redness-prone visual patterns that may be influenced by lighting, heat, or skin sensitivity.",
    suggestion:
      "A calming routine is preferred. Track possible triggers and avoid products that cause stinging or flushing.",
    use: "Gentle cleanser, calming moisturizer, daily sunscreen.",
    avoid: "Heat triggers, harsh scrubs, and irritating actives.",
  },
  Keratosis: {
    analysis:
      "This selected area falls into a visually sensitive category and should be interpreted cautiously.",
    suggestion:
      "Because this category is visually sensitive, consider professional review if this is a real skin concern.",
    use: "Document changes, use sunscreen, seek qualified review if concerned.",
    avoid: "Relying on this app for medical decisions.",
  },
  Carcinoma: {
    analysis:
      "This selected area falls into a visually sensitive category and should not be interpreted as a routine skincare concern.",
    suggestion:
      "If this reflects a real changing or unusual area on your skin, professional review is recommended.",
    use: "Document changes, use sunscreen, consult a qualified professional if concerned.",
    avoid: "Treating this visual prediction as a confirmed condition.",
  },
};

function guidanceForLabel(label, broadCategory) {
  if (CONCERN_GUIDANCE[label]) return CONCERN_GUIDANCE[label];
  if (broadCategory === "Acne-spectrum lesions") return CONCERN_GUIDANCE.Acne;
  if (broadCategory === "Milia / keratin cyst") return CONCERN_GUIDANCE.Milia;
  if (broadCategory === "Other inflammatory skin conditions") return CONCERN_GUIDANCE.Eczema;
  if (broadCategory === "Keratosis / cancer-related") return CONCERN_GUIDANCE.Keratosis;
  return {
    analysis:
      "This selected area shows a visible skin concern pattern that stands out from the surrounding skin texture.",
    suggestion:
      "Use this as a visual reference only. If the area is changing, painful, or concerning, consider professional review.",
    use: "Clear lighting, consistent skincare, sunscreen.",
    avoid: "Picking, harsh products, and using this result as treatment advice.",
  };
}

const lesionData = {
  Papular: {
    title: "Papular lesion",
    confidence: "92%",
    yolo: "Papular · 92%",
    resnet: "Not required",
    gradcam: "Grad-CAM highlights lesion centre",
    gradcamText:
      "The highlighted heatmap region overlaps with the raised inflamed area inside the YOLO bounding box, supporting the papular classification.",
    recommendation:
      "Use a gentle cleanser and consider niacinamide or benzoyl peroxide spot treatment. Avoid picking or popping the lesion.",
    use: "Low-strength benzoyl peroxide, niacinamide serum, barrier-friendly moisturiser.",
    avoid: "Harsh scrubs, squeezing, heavy occlusive products.",
  },
  Blackhead: {
    title: "Blackhead detection",
    confidence: "88%",
    yolo: "Blackhead · 88%",
    resnet: "Not required",
    gradcam: "Grad-CAM optional for high-confidence YOLO box",
    gradcamText:
      "The YOLOv8m confidence is high, so the crop does not need secondary classification. The heatmap preview still marks the local pore cluster used for explanation.",
    recommendation:
      "A comedone-focused routine may help. Consider salicylic acid and consistent cleansing while monitoring irritation.",
    use: "Salicylic acid cleanser, non-comedogenic moisturiser, daily sunscreen.",
    avoid: "Pore strips used too often, abrasive exfoliation, sleeping with makeup.",
  },
  Purulent: {
    title: "Purulent lesion",
    confidence: "93%",
    yolo: "Pustule candidate · 61%",
    resnet: "Purulent · 93%",
    gradcam: "Grad-CAM confirms bright pus-centred region",
    gradcamText:
      "YOLOv8m produced a low-confidence candidate, so the crop was sent to ResNet18. Grad-CAM concentrates on the central inflamed area, explaining the refined purulent label.",
    recommendation:
      "Inflammatory pus-filled lesions increase scarring risk. Keep the area clean and avoid manipulation.",
    use: "Benzoyl peroxide spot care, hydrocolloid patch, gentle cleansing.",
    avoid: "Popping, strong acids layered together, shared towels.",
  },
  Whitehead: {
    title: "Whitehead detection",
    confidence: "91%",
    yolo: "Whitehead candidate · 58%",
    resnet: "Whitehead · 91%",
    gradcam: "Grad-CAM focuses on closed comedone texture",
    gradcamText:
      "Because the YOLO score is below the confidence gate, the cropped lesion is refined by ResNet18. The Grad-CAM overlay shows attention on the closed comedone texture rather than surrounding skin.",
    recommendation:
      "Closed comedones often respond to consistent exfoliation and retinoid-based routines over time.",
    use: "Low-strength retinoid at night, light moisturiser, SPF in daytime.",
    avoid: "Over-cleansing, oily cosmetics, inconsistent routine changes.",
  },
};

function showView(id) {
  views.forEach((view) => view.classList.toggle("active", view.id === id));
  navItems.forEach((item) => item.classList.toggle("active", item.dataset.view === id));
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function absoluteApiUrl(path) {
  if (!path) return null;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${apiBaseUrl}${path}`;
}

function formatPercent(value) {
  if (typeof value !== "number" || Number.isNaN(value)) return "N/A";
  return `${Math.round(value * 100)}%`;
}

function getSelectedFile() {
  return selectedFile || fileInput.files?.[0] || null;
}

async function analyzeSkinImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${apiBaseUrl}/api/analyze`, {
    method: "POST",
    body: formData,
  });

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(payload?.detail || `Analysis failed with status ${response.status}`);
  }

  if (payload.visualizations) {
    payload.visualizations.uploaded_image_url = absoluteApiUrl(payload.visualizations.uploaded_image_url);
    payload.visualizations.yolo_image_url = absoluteApiUrl(payload.visualizations.yolo_image_url);
    payload.visualizations.gradcam_image_url = absoluteApiUrl(payload.visualizations.gradcam_image_url);
  }
  if (payload.resnet?.gradcam_image_url) {
    payload.resnet.gradcam_image_url = absoluteApiUrl(payload.resnet.gradcam_image_url);
  }
  (payload.detail_items || []).forEach((item) => {
    item.crop_image_url = absoluteApiUrl(item.crop_image_url);
    item.gradcam_image_url = absoluteApiUrl(item.gradcam_image_url);
  });
  return payload;
}

function setSelectedFile(file) {
  selectedFile = file || null;
  fileName.textContent = file ? file.name : "No image selected";

  if (selectedPreviewUrl) {
    URL.revokeObjectURL(selectedPreviewUrl);
    selectedPreviewUrl = null;
  }

  if (file) {
    selectedPreviewUrl = URL.createObjectURL(file);
    selectedPreview.src = selectedPreviewUrl;
    uploadPrompt.hidden = true;
    sourceChoice.hidden = true;
    uploadZone.classList.add("has-preview");
    photoReview.hidden = false;
    stopCamera();
  } else {
    selectedPreview.removeAttribute("src");
    uploadPrompt.hidden = false;
    sourceChoice.hidden = true;
    uploadZone.classList.remove("has-preview");
    uploadZone.classList.remove("camera-active");
    photoReview.hidden = true;
  }
}

function resetSelectedPhoto() {
  stopCamera();
  setSelectedFile(null);
  fileInput.value = "";
  cameraInput.value = "";
  showView("scan");
}

async function openCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    cameraInput.click();
    return;
  }

  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    cameraVideo.srcObject = cameraStream;
    cameraVideo.hidden = false;
    uploadPrompt.hidden = true;
    sourceChoice.hidden = true;
    photoReview.hidden = true;
    cameraReview.hidden = false;
    focusRing.hidden = false;
    uploadZone.classList.add("camera-active");
  } catch (_error) {
    cameraInput.click();
  }
}

function stopCamera() {
  if (cameraStream) {
    cameraStream.getTracks().forEach((track) => track.stop());
    cameraStream = null;
  }
  cameraVideo.srcObject = null;
  cameraVideo.hidden = true;
  cameraReview.hidden = true;
  uploadZone.classList.remove("camera-active");
}

function backToPhotoSourceChoice() {
  stopCamera();
  uploadPrompt.hidden = true;
  sourceChoice.hidden = false;
  photoReview.hidden = true;
}

function getCameraCropRect() {
  const videoRect = cameraVideo.getBoundingClientRect();
  const ringRect = focusRing.getBoundingClientRect();
  const videoWidth = cameraVideo.videoWidth;
  const videoHeight = cameraVideo.videoHeight;
  const elementRatio = videoRect.width / videoRect.height;
  const sourceRatio = videoWidth / videoHeight;

  let renderedWidth = videoRect.width;
  let renderedHeight = videoRect.height;
  let offsetX = 0;
  let offsetY = 0;

  if (sourceRatio > elementRatio) {
    renderedHeight = videoRect.height;
    renderedWidth = renderedHeight * sourceRatio;
    offsetX = (videoRect.width - renderedWidth) / 2;
  } else {
    renderedWidth = videoRect.width;
    renderedHeight = renderedWidth / sourceRatio;
    offsetY = (videoRect.height - renderedHeight) / 2;
  }

  const ringLeft = ringRect.left - videoRect.left;
  const ringTop = ringRect.top - videoRect.top;
  const x = ((ringLeft - offsetX) / renderedWidth) * videoWidth;
  const y = ((ringTop - offsetY) / renderedHeight) * videoHeight;
  const width = (ringRect.width / renderedWidth) * videoWidth;
  const height = (ringRect.height / renderedHeight) * videoHeight;

  const clampedX = Math.max(0, Math.min(x, videoWidth - 1));
  const clampedY = Math.max(0, Math.min(y, videoHeight - 1));
  return {
    x: clampedX,
    y: clampedY,
    width: Math.max(1, Math.min(width, videoWidth - clampedX)),
    height: Math.max(1, Math.min(height, videoHeight - clampedY)),
  };
}

function captureCameraPhoto() {
  if (!cameraStream || !cameraVideo.videoWidth) return;

  const crop = getCameraCropRect();
  cameraCanvas.width = Math.round(crop.width);
  cameraCanvas.height = Math.round(crop.height);
  const context = cameraCanvas.getContext("2d");
  context.drawImage(
    cameraVideo,
    crop.x,
    crop.y,
    crop.width,
    crop.height,
    0,
    0,
    cameraCanvas.width,
    cameraCanvas.height
  );
  cameraCanvas.toBlob((blob) => {
    if (!blob) return;
    const file = new File([blob], `camera-face-crop-${Date.now()}.jpg`, { type: "image/jpeg" });
    setSelectedFile(file);
  }, "image/jpeg", 0.92);
}

async function runScan() {
  const file = getSelectedFile();
  if (!file) {
    fileName.textContent = "Choose a JPG or PNG before running analysis.";
    showView("scan");
    return;
  }

  showView("processing");
  startAnalysisBtn.disabled = true;

  try {
    latestApiResult = await analyzeSkinImage(file);
    renderApiResult(latestApiResult);
    showView("results");
  } catch (error) {
    renderApiError(error);
    showView("results");
  } finally {
    startAnalysisBtn.disabled = false;
  }
}

function renderApiError(error) {
  document.querySelector("#reportEyebrow").textContent = "Analysis unavailable";
  document.querySelector("#reportTitle").textContent = "The backend could not complete this prediction";
  document.querySelector("#reportSubtitle").textContent = error.message;
  document.querySelector("#severityValue").textContent = "Unavailable";
  document.querySelector("#severityText").textContent =
    "Check that the FastAPI server is running and both model weight files are available.";
  document.querySelector("#areasFound").textContent = "0";
  document.querySelector("#modelSource").textContent = "Model source: Offline";
  document.querySelector("#countList").innerHTML = "";
}

function renderApiResult(result) {
  const report = result.frontend_report;
  const summary = result.final_summary;
  const visualizations = result.visualizations || {};

  document.querySelector("#reportEyebrow").textContent = "Analysis Summary";
  document.querySelector("#reportTitle").textContent = ANALYSIS_PAGE_COPY.pageTitle;
  document.querySelector("#reportSubtitle").textContent = ANALYSIS_PAGE_COPY.pageSubtitle;
  document.querySelector("#severityValue").textContent = report.severity;
  document.querySelector("#severityText").textContent = report.consumer_summary || summary.explanation;
  document.querySelector("#areasFound").textContent = String(report.areas_found);
  document.querySelector("#modelSource").textContent = `Interpretation type: ${result.visualization_type} · Model source: ${result.model_source}`;

  renderCounts(report.label_counts, report.primary_label);
  renderMainVisualization(visualizations, result);
  renderDetections(result);
  if ((result.detail_items || []).length) {
    renderDetailFromResult(result, 0);
  } else {
    renderNoSelectedConcern(result);
  }
}

function renderCounts(labelCounts, fallbackLabel) {
  const countList = document.querySelector("#countList");
  const entries = Object.entries(labelCounts || {});
  countList.innerHTML = entries
    .map(([label, count]) => `<div><span class="chip papular"></span>${label} <strong>${count}</strong></div>`)
    .join("");
}

function renderMainVisualization(visualizations, result) {
  const image = document.querySelector("#analysisImage");
  const stage = document.querySelector("#analysisStage");
  const fallback = document.querySelector(".result-face");
  const source = visualizations.uploaded_image_url || visualizations.yolo_image_url;
  if (source) {
    image.onload = () => renderDetections(result);
    image.src = source;
    stage.hidden = false;
    fallback.hidden = true;
  } else {
    stage.hidden = true;
    fallback.hidden = false;
  }
}

function renderDetections(result) {
  const overlay = document.querySelector("#detectionOverlay");
  const imageSize = result.frontend_report?.image_size;
  const items = result.detail_items || [];
  overlay.innerHTML = "";

  if (!items.length || !imageSize?.width || !imageSize?.height) {
    return;
  }

  items.forEach((item, index) => {
    const [x1, y1, x2, y2] = item.box_xyxy;
    const button = document.createElement("button");
    button.className = "detection";
    button.classList.toggle("refined", item.display_mode === "gradcam_crop");
    button.classList.toggle("whole-image", item.kind === "resnet_whole_image_gradcam");
    button.dataset.label = item.label || "Whole image";
    button.setAttribute("aria-label", `${item.label || "Whole image"} concern region`);
    button.style.left = `${(x1 / imageSize.width) * 100}%`;
    button.style.top = `${(y1 / imageSize.height) * 100}%`;
    button.style.width = `${((x2 - x1) / imageSize.width) * 100}%`;
    button.style.height = `${((y2 - y1) / imageSize.height) * 100}%`;
    button.addEventListener("click", () => {
      renderDetailFromResult(result, index);
      showView("detail");
    });
    overlay.appendChild(button);
  });
}

function renderDetailImage(item, mode) {
  const image = document.querySelector("#gradcamImage");
  const view = document.querySelector("#gradcamView");
  const source = mode === "gradcam" ? item.gradcam_image_url || item.crop_image_url : item.crop_image_url;
  view.classList.toggle("gradcam-on", mode === "gradcam" && Boolean(item.gradcam_image_url));
  if (source) {
    image.src = source;
    image.hidden = false;
  } else {
    image.hidden = true;
  }
}

function renderNoSelectedConcern(result) {
  currentDetailItem = null;
  document.querySelector("#detailTitle").textContent = "No selected concern region";
  document.querySelector("#detailConfidence").textContent = formatPercent(result.final_summary?.confidence_or_probability);
  document.querySelector("#yoloDecision").textContent = "No local box";
  document.querySelector("#resnetDecision").textContent = result.final_summary?.primary_result
    ? `${result.final_summary.primary_result} · ${formatPercent(result.final_summary.confidence_or_probability)}`
    : "Not available";
  document.querySelector("#gradcamDecision").textContent = "Not shown";
  document.querySelector("#gradcamText").textContent =
    "No local concern region was selected because the image did not return a visible detection box.";
  document.querySelector("#recommendationText").textContent =
    "Review the overall visual summary on the analysis page. There is no region-level Grad-CAM explanation for this result.";
  document.querySelector("#useText").textContent = "Use clear lighting and a centered face photo for another scan.";
  document.querySelector("#avoidText").textContent = "Avoid interpreting this as a region-level finding.";
  document.querySelector("#gradcamImage").hidden = true;
  document.querySelector("#gradcamView").classList.remove("gradcam-on");
  document.querySelectorAll(".crop-tabs button").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === "crop");
  });
}

function renderDetailFromResult(result, index = 0) {
  const item = (result.detail_items || [])[index];
  if (!item) return;

  currentDetailItem = item;
  const preferredMode = item.display_mode === "gradcam_crop" ? "gradcam" : "crop";
  document.querySelector("#detailTitle").textContent = item.title;
  document.querySelector("#detailConfidence").textContent = formatPercent(item.auxiliary_confidence ?? item.confidence);
  document.querySelector("#yoloDecision").textContent = item.yolo_decision;
  document.querySelector("#resnetDecision").textContent = item.resnet_decision;
  document.querySelector("#gradcamDecision").textContent =
    item.display_mode === "gradcam_crop" ? "Grad-CAM crop" : "Original crop";
  const guidance = guidanceForLabel(item.label, item.broad_category);
  document.querySelector("#gradcamText").textContent = guidance.analysis;
  document.querySelector("#recommendationText").textContent = guidance.suggestion;
  document.querySelector("#useText").textContent = guidance.use;
  document.querySelector("#avoidText").textContent = guidance.avoid;

  document.querySelectorAll(".crop-tabs button").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === preferredMode);
  });
  renderDetailImage(item, preferredMode);
}

function setLesion(name) {
  const data = lesionData[name] || lesionData.Papular;
  document.querySelector("#detailTitle").textContent = data.title;
  document.querySelector("#detailConfidence").textContent = data.confidence;
  document.querySelector("#yoloDecision").textContent = data.yolo;
  document.querySelector("#resnetDecision").textContent = data.resnet;
  document.querySelector("#gradcamDecision").textContent = data.gradcam;
  document.querySelector("#gradcamText").textContent = data.gradcamText;
  document.querySelector("#recommendationText").textContent = data.recommendation;
  document.querySelector("#useText").textContent = data.use;
  document.querySelector("#avoidText").textContent = data.avoid;

  document.querySelectorAll(".detection").forEach((button) => {
    button.classList.toggle("active", button.dataset.lesion === name);
  });
}

navItems.forEach((item) => {
  item.addEventListener("click", () => showView(item.dataset.view));
});

document.querySelector("#startScanTop").addEventListener("click", () => showView("scan"));
document.querySelector("#startScanHero").addEventListener("click", () => showView("scan"));
document.querySelector("#newScanBtn").addEventListener("click", resetSelectedPhoto);
document.querySelector("#backToResults").addEventListener("click", () => showView("results"));
startAnalysisBtn.addEventListener("click", runScan);
document.querySelector("#reuploadBtn").addEventListener("click", resetSelectedPhoto);
document.querySelector("#cameraBtn").addEventListener("click", openCamera);
captureBtn.addEventListener("click", captureCameraPhoto);
cameraBackBtn.addEventListener("click", backToPhotoSourceChoice);
document.querySelector("#browseBtn").addEventListener("click", () => fileInput.click());

uploadZone.addEventListener("click", (event) => {
  if (event.target.closest("button") || selectedFile || cameraStream) return;
  sourceChoice.hidden = false;
});

fileInput.addEventListener("change", () => {
  const [file] = fileInput.files;
  setSelectedFile(file);
});

cameraInput.addEventListener("change", () => {
  const [file] = cameraInput.files;
  setSelectedFile(file);
});

["dragenter", "dragover"].forEach((eventName) => {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((eventName) => {
  uploadZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    uploadZone.classList.remove("dragover");
  });
});

uploadZone.addEventListener("drop", (event) => {
  const [file] = event.dataTransfer.files;
  if (file) {
    const dataTransfer = new DataTransfer();
    dataTransfer.items.add(file);
    fileInput.files = dataTransfer.files;
  }
  setSelectedFile(file);
});

document.querySelectorAll(".detection").forEach((button) => {
  button.addEventListener("click", () => {
    if (latestApiResult) {
      renderDetailFromResult(latestApiResult);
    } else {
      setLesion(button.dataset.lesion);
    }
    showView("detail");
  });
});

document.querySelectorAll(".crop-tabs button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".crop-tabs button").forEach((tab) => {
      tab.classList.toggle("active", tab === button);
    });
    if (currentDetailItem) {
      renderDetailImage(currentDetailItem, button.dataset.mode);
    } else {
      document.querySelector("#gradcamView").classList.toggle("gradcam-on", button.dataset.mode === "gradcam");
    }
  });
});
