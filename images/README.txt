Put your night-sky images in this folder (subfolders are fine).
In the app choose  Source -> Image files  and they are loaded automatically.
Supported: .png .jpg .jpeg .tif .tiff .bmp .webp  (big photos are shrunk to 1600 px).

WHAT WORKS BEST
  * Wide field: 15 to 70 degrees across. The catalogue only goes to magnitude 6.5
    (naked-eye stars), so telescope close-ups with no bright stars cannot be solved.
  * At least ~8 clearly visible stars, sharp (not trailed), dark background.
  * No labels, grid lines, constellation art, horizon or trees covering the stars.

1) STELLARIUM SCREENSHOTS (easiest, free: https://stellarium.org  or  https://stellarium-web.org)
   Before taking the screenshot:
     - Turn off atmosphere (A) and ground (G) so the sky is black and nothing hides stars
     - Turn off constellation lines (C), names (V), art (R), grids (E / Z), and set the
       star label slider to zero in Sky and viewing options (F4)
     - Hide planets in F4 -> Solar system objects (they are not in the star catalogue
       and look like bright stars) or simply avoid parts of the sky with planets
     - Projection: leave Stellarium's default (stereographic); in the app set Lens to
       "Auto" or "Stereographic". Real camera photos use the pinhole model.
     - Zoom (mouse wheel) until the FOV in the bottom bar is between 20 and 60 degrees
     - Hide the toolbars (Ctrl+T) so the bars do not cover the image edges
   Screenshot: Ctrl+S  (saved to Pictures\Stellarium by default) -> copy the file here.
   In the app, type that FOV into the FOV box with "across height" (Stellarium's FOV is
   the vertical one; if it does not solve, try "across width" or Unknown). The solver
   also works with FOV = Unknown, it is just slower.

2) REAL PHOTOS
   - Your own phone/camera on a tripod: night mode or 5-20 s exposure, main (not ultra-wide)
     lens, point at a bright constellation (Orion, Cassiopeia, Big Dipper, Scorpius, Cygnus).
   - Wikimedia Commons (free licences): search "Orion constellation photograph",
     "Big Dipper night sky", "Cassiopeia constellation" - pick wide-field photos without
     text or drawn lines.
   - nova.astrometry.net -> "Explore" shows user images together with their solved
     RA/Dec and field size, so you can check AstraNav's answer against astrometry.net.
   Phone and wide-angle lenses have some distortion, so expect a larger fit residual
   than with Stellarium screenshots.

TIPS IF AN IMAGE IS NOT SOLVED
   - Give the FOV if you know it (even roughly, +/-25% is fine).
   - Lower "Min blob area" to 1-2 px for screenshots with tiny stars.
   - Turn on "Median background subtraction" for photos with sky glow or light pollution.
   - Look at the "Binary mask" tab: you want mostly separate star dots, not big blobs.
