// Chụp ảnh giao diện sản phẩm (Dashboard + MapView + AlertManagement) bằng Playwright.
//
// Yêu cầu: hệ thống đang chạy (tốt nhất là start.bat đầy đủ: CARLA + server --with-ai + frontend),
// để giao diện có dữ liệu thật (camera feed, track, route trên bản đồ). Nếu chạy không có CARLA,
// ảnh sẽ chỉ là khung giao diện rỗng.
//
// Cách chạy (từ thư mục component_for_final/screenshots):
//   npm init -y
//   npm i -D playwright
//   npx playwright install chromium
//   node capture_ui.mjs            # mặc định http://localhost:3000
//   BASE=http://localhost:3000 node capture_ui.mjs
//
import { chromium } from 'playwright';

const BASE = process.env.BASE || 'http://localhost:3000';
const OUT = process.env.OUT || '.';

const pages = [
  { path: '/',       name: '14_ui_dashboard_camera_grid',   wait: 4000 },
  { path: '/map',    name: '15_ui_mapview_route_prediction', wait: 5000 },
  { path: '/alerts', name: '16_ui_alert_management',         wait: 3000 },
];

const run = async () => {
  const browser = await chromium.launch();
  const ctx = await browser.newContext({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  for (const p of pages) {
    const url = BASE + p.path;
    try {
      console.log('→', url);
      await page.goto(url, { waitUntil: 'networkidle', timeout: 30000 });
      await page.waitForTimeout(p.wait); // chờ MJPEG/Leaflet/WebSocket render
      await page.screenshot({ path: `${OUT}/${p.name}.png`, fullPage: false });
      console.log('  ✓ saved', `${p.name}.png`);
    } catch (e) {
      console.error('  ✗ lỗi', url, e.message);
    }
  }
  await browser.close();
};
run();
