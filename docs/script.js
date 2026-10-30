const panelTitle = document.getElementById("panelTitle");
const panelText = document.getElementById("panelText");
const panelBullets = document.getElementById("panelBullets");
const panelFigure = document.getElementById("panelFigure");
const panelImage = document.getElementById("panelImage");
const panelCaption = document.getElementById("panelCaption");
const panelLinks = document.getElementById("panelLinks");
const clickableItems = Array.from(document.querySelectorAll("[data-id]"));

function renderPanel(id) {
  const item = CONTENT[id] || CONTENT.awetrim;
  panelTitle.textContent = item.title;
  panelText.textContent = item.text;
  panelBullets.innerHTML = item.bullets?.length
    ? `<ul>${item.bullets.map(point => `<li>${point}</li>`).join("")}</ul>`
    : "";
  if (item.image) {
    panelImage.src = item.image;
    panelImage.alt = item.title;
    panelCaption.textContent = item.caption || "";
    panelFigure.style.display = "";
  } else {
    panelFigure.style.display = "none";
  }
  panelLinks.innerHTML = item.links?.length
    ? item.links.map(link => `<a href="${link.url}" target="_blank" rel="noopener">${link.label}</a>`).join("")
    : "";

  clickableItems.forEach(element => {
    element.classList.toggle("active", element.dataset.id === id);
  });
}

clickableItems.forEach(element => {
  element.addEventListener("click", event => {
    event.stopPropagation();
    renderPanel(element.dataset.id);
  });
});

const lightbox = document.getElementById("lightbox");
const lightboxImage = document.getElementById("lightboxImage");
const lightboxCaption = document.getElementById("lightboxCaption");

panelImage.addEventListener("click", () => {
  lightboxImage.src = panelImage.src;
  lightboxImage.alt = panelImage.alt;
  lightboxCaption.textContent = panelCaption.textContent;
  lightbox.hidden = false;
});
lightbox.addEventListener("click", () => { lightbox.hidden = true; });
document.addEventListener("keydown", event => {
  if (event.key === "Escape") lightbox.hidden = true;
});

renderPanel("awetrim");
