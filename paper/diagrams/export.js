const puppeteer = require('puppeteer');
const path = require('path');

const figures = [
  { html: 'figure1_agent_domains.html', selector: '.diagram', width: 800 },
  { html: 'figure2_infrastructure.html', selector: '.diagram', width: 1100 },
  { html: 'figure3_population.html', selector: '.diagram', width: 1000 },
];

(async () => {
  const browser = await puppeteer.launch({ headless: true });

  for (const fig of figures) {
    const page = await browser.newPage();
    await page.setViewport({ width: fig.width, height: 800, deviceScaleFactor: 3 });

    const filePath = path.resolve(__dirname, fig.html);
    await page.goto('file://' + filePath, { waitUntil: 'networkidle0' });

    // Wait for fonts to load
    await page.evaluateHandle('document.fonts.ready');

    const el = await page.$(fig.selector);
    const pdfName = fig.html.replace('.html', '.pdf');
    const pngName = fig.html.replace('.html', '.png');

    // Screenshot the element tightly
    await el.screenshot({ path: path.resolve(__dirname, pngName), type: 'png' });
    console.log('Exported ' + pngName);

    await page.close();
  }

  await browser.close();
})();
