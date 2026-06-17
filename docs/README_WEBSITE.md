# AWETrim interactive framework website

This is a static clickable website for the AWETrim computational workflow.

This version does not use the uploaded PDF or a diagram image. The framework is drawn directly with HTML and CSS. It uses a grouped layout:

- Inputs
- Experimental reconstruction
- AWETrim core framework
- Outputs and applications

Each block is clickable. Clicking a block updates the information panel on the right.

## Open locally

Open `index.html` directly in your browser, or run a local server from this folder:

```bash
python -m http.server 8000
```

Then open:

```text
http://localhost:8000
```

## Add your own images

1. Put your image files in the `img/` folder.
2. Open `content.js`.
3. Find the block you want to edit.
4. Change the `image` path and `caption`.

Example:

```js
"ekf-awe": {
  title: "EKF-AWE Experimental Reconstruction",
  image: "img/ekf_reconstruction.png",
  caption: "Example reconstructed wind speed and kite states."
}
```

Recommended image formats: `.png`, `.jpg`, `.webp`, or `.svg`.

## Header logo

The site header carries the TU Delft institutional logo at the top right
(`img/tudelft-logo.svg`, linking to `tudelft.nl`). The shipped file is a
self-contained SVG placeholder in the TU Delft house colour — swap it for the
official TU Delft asset by replacing that file (keep the same name) or editing the
`src` of the `.header-logo` image in `index.html`.

## Funding band and logos

The dark-text funding band above the footer carries the MERIDIONAL logo, the
"Funded by the European Union" emblem and the Horizon Europe acknowledgment
(Grant Agreement No. 101084216).

- MERIDIONAL logo: `img/Meridional_logo.png` (colour version, for the light band).
  Swap the file or change the `src` of the `.funding-meridional` image to replace it.
- EU emblem: `img/eu-funded.svg` — a self-contained SVG (no external hotlink). The
  star ring and text colour are generated; edit the SVG directly to recolour.

The "model reduction / aero identification" arrow in the framework box is a
clickable block (`data-id="model-reduction"`) whose panel text lives in
`content.js`; it explains that the ROM is identified from the aero-structural model.

## Edit the layout

Most layout changes are in `style.css`.

Useful sections:

- `.node-grid-inputs` controls the input block layout.
- `.framework-shell` controls the AWETrim main box.
- `.model-row` controls the Aero-Structural Model and Reduced-Order Model row.
- `.support-row` controls Tether, Winch and Wind Models.
- `.node-grid-apps` controls the outputs/applications block.

## Publish on GitHub Pages

A simple setup is:

1. Copy these files into a `docs/` folder in your repository.
2. Commit and push.
3. Go to repository Settings > Pages.
4. Select the `docs/` folder as the source.
5. Save and wait for GitHub to publish the site.
