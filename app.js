const imageUpload = document.getElementById("image-upload");
const imagePreview = document.getElementById("image-preview");
const uploadPlaceholder = document.getElementById("upload-placeholder");
const clearButton = document.getElementById("clear-button");

imageUpload.addEventListener("change", function () {
  const file = this.files[0];

  if (!file) {
    return;
  }

  const imageURL = URL.createObjectURL(file);

  imagePreview.src = imageURL;
  imagePreview.style.display = "block";
  uploadPlaceholder.style.display = "none";
});

clearButton.addEventListener("click", function () {
  imageUpload.value = "";
  imagePreview.src = "";
  imagePreview.style.display = "none";
  uploadPlaceholder.style.display = "block";
});
