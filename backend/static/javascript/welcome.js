const image_input = document.querySelector("#imageUpload");
const image_display = document.querySelector("#image-display")
var uploaded_image = ""

image_input.addEventListener("change", function() {
  const reader = new FileReader();
  reader.addEventListener("load", () => {
    var image = new Image()
    image.src = reader.result
    image.onload = function() {
        image_display.setAttribute("style", `width:${image.width}px; height:${image.height}px; background-image:url(${image.src})`);
        document.getElementsByClassName("predict-btn")[0].disabled = false
        document.getElementById("predict-result").innerHTML = ""
    }
  });
  reader.readAsDataURL(this.files[0]);
});


document.getElementsByClassName("predict-btn")[0].addEventListener("click", function() {
    document.getElementsByClassName("prediction-section")[0].setAttribute("style", "display:block");
});