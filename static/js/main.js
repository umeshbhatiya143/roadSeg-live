function openModal(videoName) {
  fetch(`/process/${videoName}`)
    .then(res => res.json())
    .then(data => {
      document.getElementById("segVideo").src = data.video_url;
      document.getElementById("videoModal").style.display = "block";
    });
}

function closeModal() {
  document.getElementById("videoModal").style.display = "none";
}

function openUploadModal() {
  document.getElementById("uploadModal").style.display = "block";
}

function closeUploadModal() {
  document.getElementById("uploadModal").style.display = "none";
}

document.getElementById("uploadForm").addEventListener("submit", function(e) {
  e.preventDefault();
  const formData = new FormData(this);

  fetch("/upload", {
    method: "POST",
    body: formData
  })
    .then(res => res.json())
    .then(data => {
      document.getElementById("uploadedSegVideo").src = data.video_url;
    });
});