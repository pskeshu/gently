/**
 * U16Math — the pure arithmetic behind the uint16 viewer: a histogram of a
 * frame, ImageJ's auto contrast off it, a lookup table from black, white and gamma, and
 * the fill of an RGBA buffer through that table. No DOM, so it is tested in
 * Node (tests/js/u16-math.test.mjs) and loaded as a plain script in the page.
 */
const U16Math = {
    /** Counts per value, 65536 bins. */
    histogram(u16) {
        const h = new Uint32Array(65536);
        for (let i = 0; i < u16.length; i++) h[u16[i]]++;
        return h;
    },

    /**
     * ImageJ's auto contrast, as the Auto button of Brightness/Contrast does
     * it: ij.plugin.frame.ContrastAdjuster.autoAdjust over the 16-bit
     * histogram ij.process.ShortStatistics builds (ImageJ 1.54u; Fiji ships
     * the same class). The 65536 counts are rebinned into 256 over the data's
     * min..max with a bin size of (max - min + 1) / 256; a bin holding more
     * than a tenth of the pixels is ignored as background; black and white
     * are the first bins from each end holding more than total / autoThreshold
     * pixels, read at the bin's lower edge, as ImageJ does. autoThreshold
     * starts at 5000 and halves on each repeat, so a second press cuts
     * tighter; under 10 it is back to 5000. Pass 0 for a fresh image and the
     * returned value back in for a repeat. A flat frame, or a search that
     * finds nothing, gives the data range, as ImageJ's reset does.
     * Returns { lo, hi, autoThreshold }.
     */
    imagejAuto(hist, total, autoThreshold = 0) {
        autoThreshold = autoThreshold < 10 ? 5000 : Math.trunc(autoThreshold / 2);
        if (!total) return { lo: 0, hi: 65535, autoThreshold };
        let min = 0, max = hist.length - 1;
        while (hist[min] === 0 && min < hist.length - 1) min++;
        while (hist[max] === 0 && max > 0) max--;
        const nBins = 256, binSize = (max - min + 1) / nBins, scale = 1 / binSize;
        const h = new Uint32Array(nBins);
        for (let i = min; i <= max; i++) {
            if (hist[i]) h[Math.min(nBins - 1, Math.trunc(scale * (i - min)))] += hist[i];
        }
        const limit = Math.trunc(total / 10), threshold = Math.trunc(total / autoThreshold);
        const over = i => (h[i] > limit ? 0 : h[i]) > threshold;
        let hmin = 0;
        while (!over(hmin) && hmin < nBins - 1) hmin++;
        let hmax = nBins - 1;
        while (!over(hmax) && hmax > 0) hmax--;
        let lo = min, hi = max;
        if (hmax >= hmin) {
            lo = min + hmin * binSize; hi = min + hmax * binSize;
            if (lo === hi) { lo = min; hi = max; }
        }
        return { lo, hi, autoThreshold };
    },

    /**
     * A table from value to display grey: 0 at or below black, 255 at or above
     * white, a gamma curve between. gamma 1 is linear; below 1 lifts the dark end.
     */
    lut(black, white, gamma = 1) {
        const t = new Uint8ClampedArray(65536);
        const lo = Math.max(0, Math.min(65535, black | 0));
        const hi = Math.max(lo + 1, Math.min(65535, white | 0));
        const span = hi - lo, g = gamma > 0 ? gamma : 1;   // pow(x, γ): γ below 1 lifts the dark end
        for (let v = 0; v < 65536; v++) {
            if (v <= lo) { t[v] = 0; continue; }
            if (v >= hi) { t[v] = 255; continue; }
            t[v] = Math.round(255 * Math.pow((v - lo) / span, g));
        }
        return t;
    },

    /** Fill an RGBA byte buffer (4 per pixel) from the frame through the table. */
    shade(u16, lut, rgba) {
        for (let i = 0, j = 0; i < u16.length; i++, j += 4) {
            const g = lut[u16[i]];
            rgba[j] = g; rgba[j + 1] = g; rgba[j + 2] = g; rgba[j + 3] = 255;
        }
        return rgba;
    },

    /** The frame's shape and range from the response headers gently.core.u16 writes. */
    parseHeaders(get) {
        const n = k => { const v = get(`X-Image-${k}`); return v == null || v === '' ? null : Number(v); };
        return {
            width: n('Width'), height: n('Height'), scale: n('Scale') || 1,
            sourceWidth: n('Source-Width'), sourceHeight: n('Source-Height'),
            min: n('Min'), max: n('Max'), planes: n('Planes'), plane: n('Plane'),
        };
    },
};

if (typeof module !== 'undefined' && module.exports) module.exports = U16Math;
