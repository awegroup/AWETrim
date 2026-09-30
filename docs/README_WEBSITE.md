# AWETrim interactive framework website

This is a static clickable website for the AWETrim computational workflow.

The framework is drawn directly with HTML and CSS, in the style of the AWETrim
overview figure (`img/awetrim-overview.png`): white boxes with dark outlines inside
one rounded AWETrim frame, with arrows between them. Top to bottom:

- Inputs, feeding the AWETrim frame
- AWETrim core: VSM <-> Billow (the aero-structural coupling), "identified
  aerodynamics" down into the reduced-order model (fed by the tether, winch and
  wind system models), down into trajectory optimisation; flight-data processing
  and the shared kinematics as notes at the bottom of the frame
- Experimental reconstruction (dashed "validation and model identification"
  arrow up into the frame) and Outputs and applications (arrow out of the frame),
  side by side

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

The arrows inside the frame are clickable too: the VSM <-> Billow arrow opens the
aero-structural model (`data-id="aero-structural"`) and the "identified
aerodynamics" arrow opens the model reduction (`data-id="model-reduction"`); their
panel text lives in `content.js`.

## Edit the layout

Most layout changes are in `style.css`.

Useful sections:

- `:root` holds the diagram colours (`--frame`, `--arrow`, `--navy`, `--shell-fill`).
- `.node-grid-inputs` controls the input block layout.
- `.framework-shell` controls the AWETrim main box.
- `.solver-pair` controls the VSM <-> Billow row.
- `.rom-row` controls the system-model chips and the reduced-order model.
- `.v-arrow` / `.h-arrow` draw the arrows (`-up`, `-dashed`, `-both` variants).
- `.bottom-row` controls the reconstruction and outputs cards under the frame.
- `.node-grid-apps` controls the outputs/applications block.

## Publish on GitHub Pages

A simple setup is:

1. Copy these files into a `docs/` folder in your repository.
2. Commit and push.
3. Go to repository Settings > Pages.
4. Select the `docs/` folder as the source.
5. Save and wait for GitHub to publish the site.
