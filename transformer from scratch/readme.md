### Training Flow Diagram


<!-- ![training loop](./trainprocess.png) -->

<div style="text-align: center;">
    <img id="myImage" src="./trainprocess.png" alt="training process" style="width: 50%; max-width: 500px; transition: all 0.3s ease;">
    <br>
    <button onclick="zoomIn()">Zoom In</button>
    <button onclick="zoomOut()">Zoom Out</button>
</div>

<script>
function zoomIn() {
    var img = document.getElementById("myImage");
    var width = img.clientWidth;
    img.style.width = (width + 20) + "px";
}

function zoomOut() {
    var img = document.getElementById("myImage");
    var width = img.clientWidth;
    img.style.width = (width - 20) + "px";
}
</script>