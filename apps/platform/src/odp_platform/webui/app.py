from __future__ import annotations

import logging
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

import gradio as gr

from odp_platform.common.logging_utils import get_logger
from odp_platform.common.paths import LOGGING_DIR, ROOT_DIR
from odp_platform.webui.config_tab import create_config_ui
from odp_platform.webui.dashboard import create_dashboard_ui
from odp_platform.webui.dataset_browser import create_dataset_browser_ui
from odp_platform.webui.model_demo import create_model_demo_ui
from odp_platform.webui.training_tab import create_training_ui
from odp_platform.webui.user_tabs import (
    create_single_detection_ui,
    create_folder_detection_ui,
    create_video_detection_ui,
    create_live_camera_ui,
    create_model_selection_ui,
    create_llm_chat_ui,
)
from odp_platform.webui.validation_tab import create_validation_ui

logger = logging.getLogger(__name__)

BACKEND_URL = "http://127.0.0.1:8000"
BACKEND_DIR = ROOT_DIR / "apps" / "web-backend"

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
WALLPAPER_PATH = ASSETS_DIR / "wallpaper-prism.png"
WALLPAPER_URL = f"/gradio_api/file={WALLPAPER_PATH.as_posix()}"

APP_CSS = """
:root {
  --odp-text: #17182b;
  --odp-muted: #8d93a6;
  --odp-page: #f7f8fc;
  --odp-card: #ffffff;
  --odp-line: #e7eaf2;
  --odp-soft: #f1f3fb;
  --odp-blue: #5368f6;
  --odp-blue-soft: #eef1ff;
  --odp-purple: #8b5cf6;
  --odp-purple-soft: #f1eaff;
  --odp-pink: #db4b8f;
  --odp-pink-soft: #fdebf5;
  --odp-green: #0f766e;
  --odp-green-soft: #e7fbf5;
  --odp-yellow: #d97706;
  --odp-yellow-soft: #fff4d8;
  --odp-sidebar: 246px;
  --odp-sidebar-mini: 88px;
  --odp-topbar: 82px;
  --odp-content-pad-x: 42px;
  --odp-radius: 18px;
  --odp-page-title-width: clamp(190px, 20vw, 280px);
}

*,
*::before,
*::after {
  box-sizing: border-box !important;
}

html,
body,
gradio-app,
.gradio-container {
  min-height: 100vh !important;
  overflow-x: hidden !important;
  color: var(--odp-text) !important;
  background: var(--odp-page) !important;
  font-family: Inter, -apple-system, BlinkMacSystemFont, "SF Pro Display", "SF Pro Text", "Segoe UI", sans-serif !important;
  text-shadow: none !important;
}

gradio-app,
.gradio-container,
main.app,
main.app > .wrap,
main.app > .wrap > .contain,
main.app > .wrap > .contain > .column {
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
  margin: 0 !important;
  padding: 0 !important;
  background: transparent !important;
}

.gradio-container::before,
footer {
  display: none !important;
}

.gradio-container:has(.odp-shell) > main.app > .wrap.default.full.translucent {
  display: none !important;
  pointer-events: none !important;
}

/* ── 修复页面被右侧截断 ── */
html, body, .gradio-container, main.app, main.app > .wrap,
main.app > .wrap > .contain, main.app > .wrap > .contain > .column {
  overflow-x: hidden !important;
  max-width: 100% !important;
  width: 100% !important;
}

.odp-shell {
  position: relative !important;
  z-index: 1 !important;
  width: 100% !important;
  max-width: 100% !important;
  min-height: 100vh !important;
  margin: 0 !important;
  padding: 0 !important;
  overflow-x: hidden !important;
  background:
    radial-gradient(circle at 78% 12%, rgba(83, 104, 246, 0.08), transparent 24%),
    linear-gradient(180deg, #fbfcff 0%, var(--odp-page) 38%, #f4f6fb 100%) !important;
}

.odp-shell::before {
  content: "";
  position: fixed;
  top: 0;
  right: 0;
  left: var(--odp-sidebar);
  z-index: 16;
  height: var(--odp-topbar);
  border-bottom: 1px solid var(--odp-line);
  background: rgba(255, 255, 255, 0.9);
  box-shadow: 0 10px 28px rgba(23, 24, 43, 0.03);
  backdrop-filter: blur(14px);
  -webkit-backdrop-filter: blur(14px);
  transition: left 180ms ease;
}

/* Gradio v6 layout overrides — minimal & safe */
.tabs,
.tabitem,
.tabitem > .column,
.tabitem > .column > .form,
.contain {
  min-width: 0 !important;
}

.tabs {
  position: relative !important;
  z-index: 2 !important;
  display: block !important;
  width: auto !important;
  max-width: calc(100vw - var(--odp-sidebar) - 2px) !important;
  min-height: 100vh !important;
  margin-left: var(--odp-sidebar) !important;
  padding: 122px 24px 56px !important;
  transition: margin-left 180ms ease, padding 180ms ease;
  overflow-x: hidden !important;
  overflow-y: visible !important;
}

.tabitem {
  width: 100% !important;
  max-width: 100% !important;
  min-width: 0 !important;
}

.tabitem > .column {
  max-width: 100% !important;
  min-width: 0 !important;
  width: 100% !important;
}

.tabitem .row,
.tabitem .form {
  max-width: 100% !important;
  min-width: 0 !important;
  width: 100% !important;
}

/* ── 卡片容器：只应用于显式卡片类，不全局匹配block ── */
.odp-card-like,
.odp-row > .block,
.odp-row > .form > .block {
  color: var(--odp-text) !important;
  border: 1px solid var(--odp-line) !important;
  border-radius: var(--odp-radius) !important;
  background: var(--odp-card) !important;
  box-shadow: 0 12px 30px rgba(23, 24, 43, 0.055) !important;
  text-shadow: none !important;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
}

.odp-card-like:hover,
.odp-row > .block:hover,
.odp-row > .form > .block:hover,
.odp-card-like:focus-within,
.odp-row > .block:focus-within,
.odp-row > .form > .block:focus-within {
  outline: 0 !important;
  box-shadow: 0 12px 30px rgba(23, 24, 43, 0.055) !important;
}

.block.odp-title {
  position: fixed !important;
  top: 0 !important;
  left: 0 !important;
  z-index: 40 !important;
  width: var(--odp-sidebar) !important;
  height: var(--odp-topbar) !important;
  min-height: var(--odp-topbar) !important;
  padding: 0 22px !important;
  border: 0 !important;
  border-right: 1px solid var(--odp-line) !important;
  border-radius: 0 !important;
  background: rgba(255, 255, 255, 0.94) !important;
  box-shadow: none !important;
}

.odp-title-art {
  display: flex;
  align-items: center;
  gap: 14px;
  width: 100%;
  color: var(--odp-text);
}

.odp-logo-mark {
  position: relative;
  display: inline-block;
  flex: 0 0 36px;
  width: 36px;
  height: 36px;
  border-radius: 12px;
  background: linear-gradient(135deg, #17182b 0%, #17182b 42%, #5368f6 43%, #5368f6 100%);
  box-shadow: 0 10px 24px rgba(83, 104, 246, 0.16);
}

.odp-logo-mark::after {
  content: "";
  position: absolute;
  left: 8px;
  top: 8px;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: #ff6ead;
  box-shadow: 10px 8px 0 #54d0ff;
}

.odp-brand-copy {
  display: flex;
  min-width: 0;
  flex-direction: column;
  gap: 2px;
}

.odp-brand-copy strong {
  overflow: hidden;
  color: var(--odp-text);
  font-size: 20px;
  font-weight: 900;
  letter-spacing: 0;
  line-height: 1.1;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.odp-brand-copy small {
  overflow: hidden;
  color: var(--odp-muted);
  font-size: 11px;
  font-weight: 700;
  line-height: 1.1;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.odp-sidebar-toggle {
  position: fixed !important;
  top: 22px !important;
  left: calc(var(--odp-sidebar) - 22px) !important;
  z-index: 60 !important;
  display: grid !important;
  place-items: center !important;
  width: 44px !important;
  min-width: 44px !important;
  max-width: 44px !important;
  height: 44px !important;
  min-height: 44px !important;
  max-height: 44px !important;
  padding: 0 !important;
  border: 1px solid var(--odp-line) !important;
  border-radius: 50% !important;
  background: var(--odp-card) !important;
  color: #7d8496 !important;
  box-shadow: 0 10px 26px rgba(23, 24, 43, 0.08) !important;
  font-size: 18px !important;
  line-height: 1 !important;
  text-shadow: none !important;
  transform: none !important;
  transition: left 180ms ease, background 140ms ease, color 140ms ease, box-shadow 140ms ease !important;
}

.odp-sidebar-toggle:hover,
.odp-sidebar-toggle:focus,
.odp-sidebar-toggle:focus-visible {
  outline: 0 !important;
  background: var(--odp-blue-soft) !important;
  color: var(--odp-blue) !important;
  box-shadow: 0 10px 26px rgba(83, 104, 246, 0.16) !important;
  transform: none !important;
}

.block.odp-sidebar-toggle-wrap,
.odp-sidebar-toggle-wrap,
.odp-sidebar-toggle-wrap > div {
  width: 0 !important;
  height: 0 !important;
  min-height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  overflow: visible !important;
}

.tab-container.visually-hidden {
  display: none !important;
}

.tab-container:not(.visually-hidden),
.tab-nav,
[role="tablist"] {
  position: fixed !important;
  top: var(--odp-topbar) !important;
  bottom: 76px !important;
  left: 0 !important;
  z-index: 28 !important;
  display: flex !important;
  flex-direction: column !important;
  align-items: stretch !important;
  width: var(--odp-sidebar) !important;
  min-width: var(--odp-sidebar) !important;
  max-width: var(--odp-sidebar) !important;
  height: auto !important;
  min-height: 0 !important;
  gap: 10px !important;
  padding: 24px 14px !important;
  margin: 0 !important;
  border: 0 !important;
  border-right: 1px solid var(--odp-line) !important;
  border-radius: 0 !important;
  background: rgba(255, 255, 255, 0.94) !important;
  box-shadow: none !important;
  overflow-y: auto !important;
  overflow-x: hidden !important;
  backdrop-filter: blur(14px) !important;
  -webkit-backdrop-filter: blur(14px) !important;
  transition: width 180ms ease, min-width 180ms ease, max-width 180ms ease, padding 180ms ease !important;
}

.tab-container button,
.tab-nav button,
[role="tablist"] button[role="tab"] {
  position: relative !important;
  display: flex !important;
  align-items: center !important;
  justify-content: flex-start !important;
  gap: 14px !important;
  width: 100% !important;
  height: 58px !important;
  min-height: 58px !important;
  max-height: 58px !important;
  padding: 0 16px !important;
  border: 0 !important;
  border-radius: 14px !important;
  background: transparent !important;
  color: #767b8d !important;
  box-shadow: none !important;
  font-size: 16px !important;
  font-weight: 800 !important;
  text-shadow: none !important;
  transform: none !important;
}

.tab-container button::after,
.tab-nav button::after,
[role="tablist"] button[role="tab"]::after {
  display: none !important;
  content: none !important;
}

.tab-container button::before,
.tab-nav button::before,
[role="tablist"] button[role="tab"]::before {
  display: grid;
  place-items: center;
  flex: 0 0 34px;
  width: 34px;
  height: 34px;
  border-radius: 10px;
  background: var(--odp-blue-soft);
  color: var(--odp-blue);
  font-size: 17px;
  line-height: 1;
}

.tab-container button:nth-child(1)::before,
.tab-nav button:nth-child(1)::before,
[role="tablist"] button[role="tab"]:nth-child(1)::before {
  content: "⌂";
}

.tab-container button:nth-child(2)::before,
.tab-nav button:nth-child(2)::before,
[role="tablist"] button[role="tab"]:nth-child(2)::before {
  content: "◎";
  background: var(--odp-purple-soft);
  color: var(--odp-purple);
}

.tab-container button:nth-child(3)::before,
.tab-nav button:nth-child(3)::before,
[role="tablist"] button[role="tab"]:nth-child(3)::before {
  content: "▣";
}

.tab-container button:nth-child(4)::before,
.tab-nav button:nth-child(4)::before,
[role="tablist"] button[role="tab"]:nth-child(4)::before {
  content: "✦";
  background: var(--odp-pink-soft);
  color: var(--odp-pink);
}

.tab-container button:nth-child(5)::before,
.tab-nav button:nth-child(5)::before,
[role="tablist"] button[role="tab"]:nth-child(5)::before {
  content: "☻";
  background: var(--odp-green-soft);
  color: var(--odp-green);
}

.tab-container button:nth-child(6)::before,
.tab-nav button:nth-child(6)::before,
[role="tablist"] button[role="tab"]:nth-child(6)::before {
  content: "⚙";
  background: var(--odp-yellow-soft);
  color: var(--odp-yellow);
}

.tab-container button:hover,
.tab-nav button:hover,
[role="tablist"] button[role="tab"]:hover,
.tab-container button.selected,
.tab-nav button.selected,
[role="tablist"] button[role="tab"].selected,
.tab-container button[aria-selected="true"],
.tab-nav button[aria-selected="true"],
[role="tablist"] button[role="tab"][aria-selected="true"] {
  background: var(--odp-blue-soft) !important;
  color: var(--odp-blue) !important;
  box-shadow: none !important;
  transform: none !important;
}

.odp-mode-bar {
  position: fixed !important;
  left: 0 !important;
  bottom: 0 !important;
  z-index: 30 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: flex-end !important;
  width: var(--odp-sidebar) !important;
  height: 76px !important;
  padding: 12px 14px !important;
  margin: 0 !important;
  border: 0 !important;
  border-top: 1px solid var(--odp-line) !important;
  border-right: 1px solid var(--odp-line) !important;
  border-radius: 0 !important;
  background: rgba(255, 255, 255, 0.94) !important;
  box-shadow: none !important;
  backdrop-filter: blur(14px) !important;
  -webkit-backdrop-filter: blur(14px) !important;
  transition: width 180ms ease, padding 180ms ease !important;
}

.odp-mode-bar > div,
.odp-mode-bar > .form,
.odp-mode-bar .block,
.odp-mode-bar .html-container,
.odp-mode-bar .prose {
  display: flex !important;
  justify-content: flex-end !important;
  width: 100% !important;
  min-height: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  overflow: visible !important;
}

.odp-mode-bar button,
.odp-gear-button,
button.odp-gear-button {
  flex: 0 0 48px !important;
  width: 48px !important;
  min-width: 48px !important;
  max-width: 48px !important;
  height: 48px !important;
  min-height: 48px !important;
  max-height: 48px !important;
  padding: 0 !important;
  margin-left: auto !important;
  border: 1px solid #fde7b0 !important;
  border-radius: 14px !important;
  background: var(--odp-yellow-soft) !important;
  color: var(--odp-yellow) !important;
  box-shadow: none !important;
  text-shadow: none !important;
  transform: none !important;
  font-size: 20px !important;
  line-height: 1 !important;
}

.odp-mode-bar button:hover,
.odp-gear-button:hover,
button.odp-gear-button:hover {
  background: var(--odp-blue-soft) !important;
  color: var(--odp-blue) !important;
  transform: none !important;
}

.odp-gear-button {
  font-family: "SF Pro Text", "Segoe UI Symbol", Arial, sans-serif !important;
}

html.odp-sidebar-collapsed .odp-shell::before,
.odp-shell.odp-sidebar-collapsed::before {
  left: var(--odp-sidebar-mini);
}

html.odp-sidebar-collapsed .block.odp-title,
html.odp-sidebar-collapsed .tab-container:not(.visually-hidden),
html.odp-sidebar-collapsed .tab-nav,
html.odp-sidebar-collapsed [role="tablist"],
html.odp-sidebar-collapsed .odp-mode-bar,
.odp-shell.odp-sidebar-collapsed .block.odp-title,
.odp-shell.odp-sidebar-collapsed .tab-container:not(.visually-hidden),
.odp-shell.odp-sidebar-collapsed .tab-nav,
.odp-shell.odp-sidebar-collapsed [role="tablist"],
.odp-shell.odp-sidebar-collapsed .odp-mode-bar {
  width: var(--odp-sidebar-mini) !important;
  min-width: var(--odp-sidebar-mini) !important;
  max-width: var(--odp-sidebar-mini) !important;
}

html.odp-sidebar-collapsed .block.odp-title,
.odp-shell.odp-sidebar-collapsed .block.odp-title {
  padding: 0 18px !important;
}

html.odp-sidebar-collapsed .odp-brand-copy,
.odp-shell.odp-sidebar-collapsed .odp-brand-copy {
  display: none !important;
}

html.odp-sidebar-collapsed .odp-title-art,
html.odp-sidebar-collapsed .odp-mode-bar,
.odp-shell.odp-sidebar-collapsed .odp-title-art,
.odp-shell.odp-sidebar-collapsed .odp-mode-bar {
  justify-content: center !important;
}

html.odp-sidebar-collapsed .odp-sidebar-toggle,
.odp-shell.odp-sidebar-collapsed .odp-sidebar-toggle {
  left: calc(var(--odp-sidebar-mini) - 22px) !important;
}

html.odp-sidebar-collapsed .tab-container:not(.visually-hidden),
html.odp-sidebar-collapsed .tab-nav,
html.odp-sidebar-collapsed [role="tablist"],
.odp-shell.odp-sidebar-collapsed .tab-container:not(.visually-hidden),
.odp-shell.odp-sidebar-collapsed .tab-nav,
.odp-shell.odp-sidebar-collapsed [role="tablist"] {
  padding: 24px 14px !important;
}

html.odp-sidebar-collapsed .tab-container button,
html.odp-sidebar-collapsed .tab-nav button,
html.odp-sidebar-collapsed [role="tablist"] button[role="tab"],
.odp-shell.odp-sidebar-collapsed .tab-container button,
.odp-shell.odp-sidebar-collapsed .tab-nav button,
.odp-shell.odp-sidebar-collapsed [role="tablist"] button[role="tab"] {
  justify-content: center !important;
  padding: 0 !important;
  font-size: 0 !important;
}

html.odp-sidebar-collapsed .tab-container button::before,
html.odp-sidebar-collapsed .tab-nav button::before,
html.odp-sidebar-collapsed [role="tablist"] button[role="tab"]::before,
.odp-shell.odp-sidebar-collapsed .tab-container button::before,
.odp-shell.odp-sidebar-collapsed .tab-nav button::before,
.odp-shell.odp-sidebar-collapsed [role="tablist"] button[role="tab"]::before {
  margin: 0 !important;
  font-size: 17px !important;
}

html.odp-sidebar-collapsed .odp-mode-bar button,
html.odp-sidebar-collapsed .odp-gear-button,
.odp-shell.odp-sidebar-collapsed .odp-mode-bar button,
.odp-shell.odp-sidebar-collapsed .odp-gear-button {
  margin: 0 auto !important;
}

html.odp-sidebar-collapsed .tabs,
html.odp-sidebar-collapsed .odp-admin-head,
.odp-shell.odp-sidebar-collapsed .tabs,
.odp-shell.odp-sidebar-collapsed .odp-admin-head {
  width: auto !important;
  margin-left: var(--odp-sidebar-mini) !important;
}

.odp-row {
  display: grid !important;
  width: 100% !important;
  gap: 20px !important;
  align-items: stretch !important;
  margin: 0 0 20px !important;
}

.odp-row > * {
  min-width: 0 !important;
  width: 100% !important;
}

.odp-row > .form {
  display: contents !important;
}

.odp-row > .form > * {
  min-width: 0 !important;
  width: 100% !important;
}

.odp-row > .block,
.odp-row > .form > .block,
.odp-row > button,
.odp-row > .form > button {
  min-height: 98px !important;
}

.odp-row-action {
  grid-template-columns: minmax(220px, 0.65fr) minmax(360px, 2fr) !important;
}

/* ── 模式切换 Radio ── */
.odp-mode-radio {
  display: flex !important;
  gap: 4px !important;
  margin-bottom: 8px !important;
  padding: 4px !important;
  background: var(--odp-bg-soft) !important;
  border-radius: 12px !important;
  border: 1px solid var(--odp-line) !important;
  width: fit-content !important;
}
.odp-mode-radio label {
  flex: 1 !important;
  padding: 8px 24px !important;
  margin: 0 !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  cursor: pointer !important;
  text-align: center !important;
  background: transparent !important;
  color: var(--odp-text-muted) !important;
  transition: all 0.15s ease !important;
}
.odp-mode-radio label:has(input:checked) {
  background: var(--odp-blue) !important;
  color: #fff !important;
  box-shadow: 0 2px 8px rgba(58, 104, 226, 0.3) !important;
}
.odp-mode-radio input[type="radio"] {
  display: none !important;
}

.odp-row-two {
  grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
}

.odp-row-three {
  grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
}

.odp-row-four {
  grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
}

.odp-row-five {
  grid-template-columns: repeat(5, minmax(0, 1fr)) !important;
}

/* ── 普通按钮 ── */
.odp-row button,
.odp-admin-layer button,
.odp-user-layer button {
  color: var(--odp-text) !important;
  border: 1px solid var(--odp-line) !important;
  border-radius: 14px !important;
  background: var(--odp-card) !important;
  box-shadow: 0 8px 20px rgba(23, 24, 43, 0.04) !important;
  font-weight: 800 !important;
  text-shadow: none !important;
  transform: none !important;
}

.odp-row button:hover,
.odp-row button:focus,
.odp-admin-layer button:hover,
.odp-user-layer button:hover {
  outline: 0 !important;
  background: var(--odp-blue-soft) !important;
  color: var(--odp-blue) !important;
  box-shadow: 0 8px 20px rgba(83, 104, 246, 0.08) !important;
  transform: none !important;
}

button.primary,
button.primary:hover,
.primary button,
.primary button:hover {
  border-color: var(--odp-blue) !important;
  background: var(--odp-blue) !important;
  color: #ffffff !important;
}

.label-wrap,
.label-wrap *,
.block label,
.block span,
.panel label,
.panel span,
.prose,
.markdown,
.markdown p {
  color: var(--odp-text) !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  text-shadow: none !important;
}

.label-wrap {
  padding: 14px 18px 4px !important;
}

input,
textarea,
select,
.input-container,
.container.show_textbox_border {
  color: var(--odp-text) !important;
  border: 1px solid transparent !important;
  border-radius: 14px !important;
  background: var(--odp-soft) !important;
  box-shadow: none !important;
  text-shadow: none !important;
}

.wrap,
.wrap-inner,
.secondary-wrap {
  border-color: transparent !important;
  background: transparent !important;
  box-shadow: none !important;
}

input:focus,
textarea:focus,
select:focus,
button:focus,
button:focus-visible {
  outline: 0 !important;
}

textarea {
  resize: none !important;
}

ul.options,
ul.options[role="listbox"] {
  z-index: 10000 !important;
  max-height: min(280px, 42vh) !important;
  margin-top: 8px !important;
  padding: 8px !important;
  transform: none !important;
  border: 1px solid var(--odp-line) !important;
  border-radius: 14px !important;
  background: #ffffff !important;
  box-shadow: 0 18px 44px rgba(23, 24, 43, 0.14) !important;
  color: var(--odp-text) !important;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
}

ul.options li,
ul.options [role="option"] {
  min-height: 36px !important;
  padding: 8px 10px !important;
  border-radius: 10px !important;
  color: var(--odp-text) !important;
  white-space: nowrap !important;
  word-break: keep-all !important;
  writing-mode: horizontal-tb !important;
  text-shadow: none !important;
}

ul.options li:hover,
ul.options [role="option"]:hover,
ul.options li[aria-selected="true"],
ul.options [role="option"][aria-selected="true"] {
  background: var(--odp-blue-soft) !important;
  color: var(--odp-blue) !important;
}

.image-container,
.upload-container,
.image-container *,
.upload-container * {
  outline: 0 !important;
  box-shadow: none !important;
}

.image-container,
.upload-container {
  background: transparent !important;
}

.dataframe,
.table-wrap,
table,
thead,
tbody,
tr,
td,
th {
  color: var(--odp-text) !important;
  background: transparent !important;
  box-shadow: none !important;
  text-shadow: none !important;
}

th,
td {
  border-color: var(--odp-line) !important;
}

/* ── 表格强制白底黑字（覆盖Gradio深色主题）── */
div.dataframe,
div.table-wrap,
div.dataframe table,
div.table-wrap table,
div.dataframe table thead,
div.dataframe table tbody,
div.dataframe table tr,
div.dataframe table td,
div.dataframe table th,
div.table-wrap table thead,
div.table-wrap table tbody,
div.table-wrap table tr,
div.table-wrap table td,
div.table-wrap table th {
  color: #17182b !important;
  background: #ffffff !important;
  border-color: #e7eaf2 !important;
}

div.dataframe table td,
div.dataframe table th,
div.table-wrap table td,
div.table-wrap table th {
  color: #17182b !important;
  background-color: #ffffff !important;
  font-weight: 600 !important;
  font-size: 14px !important;
  padding: 10px 14px !important;
  border-bottom: 1px solid #e7eaf2 !important;
}

div.dataframe table tr:nth-child(even) td {
  background-color: #f0f2f8 !important;
}

div.dataframe table tr td:first-child {
  font-weight: 700 !important;
  color: #17182b !important;
  background-color: #eef1ff !important;
}

div.dataframe {
  border: 1px solid #e7eaf2 !important;
  border-radius: 14px !important;
  overflow: hidden !important;
  background: #ffffff !important;
}

/* Chatbot 强制白底（覆盖Gradio 6.0深色主题） */
div.chatbot,
div[class*="chatbot"],
div[class*="Chatbot"],
div.chatbot > div,
div[class*="chatbot"] > div,
div[class*="Chatbot"] > div,
div.chatbot .wrap,
div.chatbot .message-wrap,
div.chatbot .bot,
div.chatbot .user,
div.chatbot [class*="message"],
div.chatbot [class*="Message"],
div.chatbot .bot *,
div.chatbot .user *,
div.chatbot [class*="bot"] *,
div.chatbot [class*="user"] *,
.chatbot,
[class*="chatbot"],
[class*="Chatbot"],
.chatbot * {
  background: #ffffff !important;
  color: #17182b !important;
  border-color: #e7eaf2 !important;
}

.table-wrap th button,
.table-container button,
.dataframe button {
  border: 0 !important;
  background: transparent !important;
  color: var(--odp-text) !important;
  box-shadow: none !important;
  transform: none !important;
}

.odp-user-head,
.odp-admin-head {
  position: relative !important;
  z-index: 2 !important;
  display: grid !important;
  gap: 14px !important;
  align-items: center !important;
  width: auto !important;
  max-width: none !important;
  margin-left: var(--odp-sidebar) !important;
  padding: 72px 24px 8px !important;
  transition: margin-left 180ms ease, padding 180ms ease !important;
}

.odp-user-head {
  grid-template-columns: var(--odp-page-title-width) minmax(0, 1fr) !important;
}

.odp-admin-head {
  grid-template-columns: var(--odp-page-title-width) minmax(150px, auto) !important;
  justify-content: space-between !important;
}

.odp-admin-head > .block:last-child {
  justify-self: end !important;
  width: auto !important;
}

.odp-user-title,
.odp-admin-title {
  width: var(--odp-page-title-width) !important;
  max-width: var(--odp-page-title-width) !important;
  height: 44px !important;
  min-height: 44px !important;
  max-height: 44px !important;
  display: flex !important;
  align-items: center !important;
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  color: var(--odp-text) !important;
  font-size: 18px !important;
  font-weight: 900 !important;
}

.odp-user-title-card,
.odp-admin-title-card {
  display: inline-flex !important;
  align-items: center !important;
  gap: 10px !important;
  width: 100% !important;
  min-height: 44px !important;
  padding: 0 !important;
  color: var(--odp-text) !important;
}

.odp-user-title-card::before,
.odp-admin-title-card::before {
  content: "";
  flex: 0 0 6px !important;
  width: 6px !important;
  height: 26px !important;
  border-radius: 999px !important;
  background: linear-gradient(180deg, var(--odp-blue), #7c5cff) !important;
  box-shadow: 0 8px 18px rgba(83, 104, 246, 0.18) !important;
}

.odp-user-title-card span,
.odp-admin-title-card span {
  display: block !important;
  overflow: hidden !important;
  font-size: 20px !important;
  line-height: 1 !important;
  font-weight: 950 !important;
  letter-spacing: 0 !important;
  text-overflow: ellipsis !important;
  white-space: nowrap !important;
}

.odp-user-title-card small,
.odp-admin-title-card small {
  display: block !important;
  overflow: hidden !important;
  margin-top: 4px !important;
  color: var(--odp-muted) !important;
  font-size: 10.5px !important;
  line-height: 1.1 !important;
  font-weight: 800 !important;
  text-overflow: ellipsis !important;
  white-space: nowrap !important;
}

.odp-user-layer .tabs,
.odp-admin-layer .tabs {
  min-height: calc(100vh - 154px) !important;
  padding-top: 8px !important;
  padding-bottom: 36px !important;
}

.odp-admin-layer .tabitem {
  padding-top: 0 !important;
}

.odp-admin-layer .odp-row {
  gap: 14px !important;
  margin-bottom: 14px !important;
}

.odp-admin-layer .odp-row > .block,
.odp-admin-layer .odp-row > .form > .block,
.odp-admin-layer .odp-row > button,
.odp-admin-layer .odp-row > .form > button {
  min-height: 92px !important;
}

.odp-admin-layer .odp-row > .block:has(.input-container),
.odp-admin-layer .odp-row > .form > .block:has(.input-container),
.odp-admin-layer .odp-row > .block:has(input),
.odp-admin-layer .odp-row > .form > .block:has(input),
.odp-admin-layer .odp-row > .block:has(select),
.odp-admin-layer .odp-row > .form > .block:has(select),
.odp-admin-layer .odp-row > .block:has([role="radiogroup"]),
.odp-admin-layer .odp-row > .form > .block:has([role="radiogroup"]) {
  min-height: 112px !important;
}

.odp-admin-layer .block,
.odp-admin-layer .panel {
  border-radius: 16px !important;
}

.odp-admin-layer .label-wrap {
  padding: 10px 14px 2px !important;
}

.odp-admin-layer input,
.odp-admin-layer textarea,
.odp-admin-layer select,
.odp-admin-layer .input-container,
.odp-admin-layer .container.show_textbox_border {
  min-height: 48px !important;
  border-radius: 12px !important;
}

.odp-admin-layer button {
  min-height: 52px !important;
  border-radius: 14px !important;
}

.odp-admin-modal {
  position: fixed !important;
  inset: 0 !important;
  z-index: 2147483647 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  padding: 24px !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: rgba(248, 250, 255, 0.85) !important;
  box-shadow: none !important;
  backdrop-filter: none !important;
  -webkit-backdrop-filter: none !important;
  flex: 0 0 auto !important;
  min-width: 0 !important;
  max-width: none !important;
  min-height: 100vh !important;
  height: 100vh !important;
  width: 100vw !important;
  margin: 0 !important;
  top: 0 !important;
  left: 0 !important;
  right: 0 !important;
  bottom: 0 !important;
}

.odp-admin-modal.odp-modal-hidden {
  display: none !important;
}

.odp-modal-card {
  width: 360px !important;
  max-width: calc(100vw - 48px) !important;
  padding: 22px !important;
  border: 1px solid var(--odp-line) !important;
  border-radius: 18px !important;
  background: #ffffff !important;
  box-shadow: 0 24px 70px rgba(23, 24, 43, 0.14) !important;
}

.odp-modal-card .block,
.odp-modal-card .form,
.odp-modal-card .panel {
  padding: 0 !important;
  border: 0 !important;
  background: transparent !important;
  box-shadow: none !important;
  min-height: 0 !important;
}

.odp-admin-login,
.odp-admin-login * {
  box-sizing: border-box !important;
}

.odp-admin-login {
  width: 100% !important;
  background: #ffffff !important;
  color: var(--odp-text) !important;
}

.odp-modal-card .html-container,
.odp-modal-card .prose,
.odp-admin-login-title,
.odp-admin-form-field,
.odp-admin-form-field span,
.odp-admin-error-text,
.odp-modal-actions {
  background: #ffffff !important;
}

.odp-admin-login-title {
  margin-bottom: 16px !important;
}

.odp-admin-login-title strong {
  display: block !important;
  font-size: 19px !important;
  line-height: 1.15 !important;
  font-weight: 900 !important;
  letter-spacing: 0 !important;
}

.odp-admin-login-title small {
  display: block !important;
  margin-top: 5px !important;
  color: var(--odp-muted) !important;
  font-size: 12px !important;
  line-height: 1.2 !important;
  font-weight: 700 !important;
}

.odp-admin-form-field {
  display: block !important;
  margin-bottom: 12px !important;
}

.odp-admin-form-field span {
  display: block !important;
  margin-bottom: 7px !important;
  color: var(--odp-text) !important;
  font-size: 13px !important;
  line-height: 1.1 !important;
  font-weight: 800 !important;
}

.odp-admin-password-input {
  width: 100% !important;
  min-height: 46px !important;
  padding: 0 14px !important;
  border: 1px solid var(--odp-line) !important;
  border-radius: 12px !important;
  background: #ffffff !important;
  color: var(--odp-text) !important;
  box-shadow: 0 8px 18px rgba(23, 24, 43, 0.04) !important;
  font-size: 15px !important;
  font-weight: 700 !important;
}

.odp-admin-password-input::placeholder {
  color: #9aa1b2 !important;
  font-weight: 700 !important;
}

.odp-admin-error-text {
  min-height: 18px !important;
  margin: -2px 0 12px !important;
  color: var(--odp-pink) !important;
  font-size: 12px !important;
  font-weight: 800 !important;
  line-height: 1.2 !important;
  opacity: 0 !important;
}

.odp-admin-error-text.is-visible {
  opacity: 1 !important;
}

.odp-modal-card > div,
.odp-modal-card > .form {
  margin-bottom: 8px !important;
}

.odp-modal-card .label-wrap {
  padding: 0 0 4px !important;
  font-weight: 700 !important;
  font-size: 13px !important;
}

.odp-modal-card input,
.odp-modal-card textarea {
  min-height: 44px !important;
  padding: 0 14px !important;
  border: 1px solid var(--odp-line) !important;
  border-radius: 12px !important;
  background: #ffffff !important;
  color: #17182b !important;
  font-size: 16px !important;
}

.odp-modal-card button {
  min-height: 44px !important;
  padding: 0 20px !important;
  border-radius: 12px !important;
  font-weight: 700 !important;
}

.odp-modal-card button:not(.primary) {
  background: #f1f3fb !important;
  color: #17182b !important;
  border: 1px solid var(--odp-line) !important;
}

.odp-modal-card button.primary,
.odp-modal-card .primary button {
  background: #5368f6 !important;
  color: #ffffff !important;
  border: 0 !important;
}

.odp-modal-actions {
  display: flex !important;
  gap: 12px !important;
  justify-content: flex-end !important;
  margin-top: 4px !important;
}

.odp-modal-actions button {
  flex: 1 !important;
}

.odp-admin-cancel-btn,
.odp-admin-confirm-btn {
  min-height: 44px !important;
  padding: 0 18px !important;
  border-radius: 12px !important;
  font-size: 15px !important;
  font-weight: 800 !important;
  cursor: pointer !important;
}

.odp-admin-cancel-btn {
  border: 1px solid var(--odp-line) !important;
  background: #f1f3fb !important;
  color: var(--odp-text) !important;
}

.odp-admin-confirm-btn {
  border: 0 !important;
  background: var(--odp-blue) !important;
  color: #ffffff !important;
}

/* ── 文件夹检测结果区：让明细表有足够宽度 ── */
.odp-result-row {
  grid-template-columns: minmax(260px, 0.72fr) minmax(560px, 1.28fr) !important;
  align-items: start !important;
}

.odp-result-row div.dataframe {
  overflow-x: auto !important;
  overflow-y: hidden !important;
}

.odp-result-row div.dataframe table,
.odp-result-row div.table-wrap table {
  width: 100% !important;
  min-width: 100% !important;
  table-layout: auto !important;
}

.odp-result-row div.dataframe table th,
.odp-result-row div.dataframe table td,
.odp-result-row div.table-wrap table th,
.odp-result-row div.table-wrap table td {
  white-space: nowrap !important;
  padding: 9px 12px !important;
  font-size: 13px !important;
}

/* ── Agent 工具开关 ── */
.odp-agent-toggle label {
  display: flex !important;
  align-items: center !important;
  gap: 10px !important;
  padding: 10px 16px !important;
  border-radius: 10px !important;
  background: #e8f5e9 !important;
  border: 2px solid #4caf50 !important;
  font-weight: 700 !important;
  font-size: 15px !important;
  cursor: pointer !important;
  transition: all 0.2s !important;
}
.odp-agent-toggle input[type="checkbox"] {
  width: 20px !important;
  height: 20px !important;
  accent-color: #2e7d32 !important;
  cursor: pointer !important;
}
.odp-agent-toggle label:has(input:not(:checked)) {
  background: #f5f5f5 !important;
  border-color: #bdbdbd !important;
}

@media (max-width: 1180px) {
  .odp-row-action,
  .odp-row-three,
  .odp-row-four,
  .odp-row-five {
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
  }
}

@media (max-width: 980px) {
  .odp-result-row {
    grid-template-columns: 1fr !important;
  }
}

@media (max-width: 820px) {
  .tabs,
  .odp-user-head,
  .odp-admin-head {
    margin-left: var(--odp-sidebar-mini) !important;
    width: auto !important;
    padding-left: 22px !important;
    padding-right: 22px !important;
  }

  .block.odp-title,
  .tab-container:not(.visually-hidden),
  .tab-nav,
  [role="tablist"],
  .odp-mode-bar {
    width: var(--odp-sidebar-mini) !important;
    min-width: var(--odp-sidebar-mini) !important;
    max-width: var(--odp-sidebar-mini) !important;
  }

  .odp-brand-copy {
    display: none !important;
  }

  .odp-sidebar-toggle {
    left: calc(var(--odp-sidebar-mini) - 22px) !important;
  }

  .odp-row,
  .odp-row-action,
  .odp-row-two,
  .odp-row-three,
  .odp-row-four,
  .odp-row-five,
  .odp-result-row,
  .odp-user-head,
  .odp-admin-head {
    grid-template-columns: 1fr !important;
  }
}

/* Prevents layout shift from scrollbar */
html { scrollbar-gutter: stable; }

/* Layer visibility control - never use Gradio visible= toggle */
.odp-layer-hidden { display: none !important; }

/* Force all tab content containers to not overflow right */
.tabitem .column:not(.odp-admin-modal),
.tabitem .form,
.tabitem .row,
.tabitem .group,
.tabitem > div,
.tabitem [class*="Column"],
.tabitem [class*="Row"] {
  max-width: 100% !important;
  width: 100% !important;
  min-width: 0 !important;
  overflow-x: hidden !important;
  overflow-wrap: break-word !important;
  word-break: break-word !important;
}

/* Ensure Gradio tabs panel doesn't overflow */
.tabitem {
  overflow-x: hidden !important;
  max-width: 100% !important;
}

/* Fix deeply nested Gradio containers */
.contain,
.wrap {
  max-width: 100% !important;
  overflow-x: hidden !important;
}

/* JS layer control: hidden by default */
.odp-layer-hidden { display: none !important; }

/* ── 强制白底（覆盖Gradio 6.0深色主题） ── */
body,
html,
.gradio-container,
.gradio-container *,
.dark,
.dark *,
[data-testid="dark"],
[class*="dark"],
body.dark,
.gradio-container.dark {
  color-scheme: light !important;
  background: var(--odp-page) !important;
  color: var(--odp-text) !important;
  --background-fill-primary: #ffffff !important;
  --background-fill-secondary: #f7f8fc !important;
  --border-color-primary: #e7eaf2 !important;
  --block-background-fill: #ffffff !important;
  --block-border-color: #e7eaf2 !important;
  --input-background-fill: #f1f3fb !important;
  --input-border-color: transparent !important;
}

/* ── 禁止gradio图片全屏预览（不影响其他交互） ── */
.image-container img {
  max-width: 100% !important; max-height: 65vh !important;
  width: auto !important; height: auto !important;
  object-fit: contain !important;
  cursor: default !important;
}
[data-testid="image-lightbox"] {
  display: none !important;
}
button[aria-label="全屏"],
button[aria-label="Fullscreen"],
button:has(svg[data-testid="FullscreenIcon"]) {
  display: none !important;
}

/* ─────────────────────────────────────────────
   Fix: 顶部参数栏 Slider 数值输入框被遮挡
   关键点：Gradio Slider 内部的 number input / reset button 被全局 input/button 高度规则污染。
   这里只针对顶部阈值滑条做局部重置。
───────────────────────────────────────────── */
.odp-param-row {
  grid-template-columns: repeat(2, minmax(360px, 1fr)) !important;
  overflow: visible !important;
  align-items: stretch !important;
}

.odp-param-row,
.odp-param-row > *,
.odp-param-row > .form,
.odp-param-row > .form > *,
.odp-param-row .block,
.odp-param-row .form,
.odp-param-row .label-wrap,
.odp-param-row .wrap,
.odp-param-row .wrap-inner,
.odp-param-row .input-container,
.odp-param-row .secondary-wrap {
  min-width: 0 !important;
  max-width: 100% !important;
  overflow: visible !important;
}

.odp-param-row > .block,
.odp-param-row > .form > .block,
.odp-param-row > button,
.odp-param-row > .form > button {
  min-height: 106px !important;
}

/* Dropdown / 刷新按钮可以保持卡片样式 */
.odp-param-row > button,
.odp-param-row > .form > button {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
}

/* 只修 Confidence / IoU 两个阈值滑条 */
.odp-threshold-slider,
.odp-param-row .block.odp-threshold-slider,
.odp-param-row .block:has(input[type="range"]) {
  min-width: 360px !important;
  min-height: 106px !important;
  overflow: hidden !important;
}

.odp-threshold-slider *,
.odp-param-row .block.odp-threshold-slider *,
.odp-param-row .block:has(input[type="range"]) * {
  box-sizing: border-box !important;
  overflow-wrap: normal !important;
  word-break: keep-all !important;
  white-space: nowrap !important;
}

/* Slider 顶部：左边 label，右边数值框 */
.odp-threshold-slider .label-wrap,
.odp-param-row .block.odp-threshold-slider .label-wrap,
.odp-param-row .block:has(input[type="range"]) .label-wrap {
  display: flex !important;
  align-items: center !important;
  justify-content: space-between !important;
  gap: 12px !important;
  height: 40px !important;
  min-height: 40px !important;
  max-height: 40px !important;
  padding: 8px 14px 0 !important;
  overflow: visible !important;
}

.odp-threshold-slider .label-wrap label,
.odp-param-row .block.odp-threshold-slider .label-wrap label,
.odp-param-row .block:has(input[type="range"]) .label-wrap label {
  min-width: 0 !important;
  flex: 1 1 auto !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

/* 重点：撤销全局 input 的 48px 高度，否则 number input 会顶坏 Slider */
.odp-threshold-slider input[type="number"],
.odp-param-row .block.odp-threshold-slider input[type="number"],
.odp-param-row .block:has(input[type="range"]) input[type="number"] {
  display: inline-block !important;
  width: 78px !important;
  min-width: 78px !important;
  max-width: 78px !important;
  height: 30px !important;
  min-height: 30px !important;
  max-height: 30px !important;
  padding: 0 8px !important;
  margin: 0 !important;
  border-radius: 10px !important;
  line-height: 30px !important;
  text-align: center !important;
  font-size: 14px !important;
  font-weight: 800 !important;
  flex: 0 0 78px !important;
}

/* 重点：撤销全局 input 的 48px 高度，否则 range 轨道会被撑乱 */
.odp-threshold-slider input[type="range"],
.odp-param-row .block.odp-threshold-slider input[type="range"],
.odp-param-row .block:has(input[type="range"]) input[type="range"] {
  height: 18px !important;
  min-height: 18px !important;
  max-height: 18px !important;
  padding: 0 !important;
  margin: 0 !important;
  flex: 1 1 auto !important;
  min-width: 180px !important;
  width: auto !important;
  background: transparent !important;
}

/* 重点：Gradio Slider 的 reset 小按钮会被全局 button min-height:52px 污染；这里隐藏它 */
.odp-threshold-slider button,
.odp-param-row .block.odp-threshold-slider button,
.odp-param-row .block:has(input[type="range"]) button {
  display: none !important;
}

/* Slider 主体区域 */
.odp-threshold-slider .wrap,
.odp-threshold-slider .wrap-inner,
.odp-threshold-slider .secondary-wrap,
.odp-param-row .block.odp-threshold-slider .wrap,
.odp-param-row .block.odp-threshold-slider .wrap-inner,
.odp-param-row .block.odp-threshold-slider .secondary-wrap,
.odp-param-row .block:has(input[type="range"]) .wrap,
.odp-param-row .block:has(input[type="range"]) .wrap-inner,
.odp-param-row .block:has(input[type="range"]) .secondary-wrap {
  min-height: 48px !important;
  height: 48px !important;
  display: flex !important;
  align-items: center !important;
  gap: 14px !important;
  padding: 0 14px 10px !important;
  overflow: visible !important;
}

/* min / max 文本禁止断行，避免 0.01 被拆成两行 */
.odp-threshold-slider span,
.odp-param-row .block.odp-threshold-slider span,
.odp-param-row .block:has(input[type="range"]) span {
  white-space: nowrap !important;
  word-break: keep-all !important;
  overflow-wrap: normal !important;
  min-width: fit-content !important;
}

@media (min-width: 2100px) {
  .odp-param-row {
    grid-template-columns:
      minmax(280px, 1fr)
      minmax(140px, 0.45fr)
      minmax(390px, 1.2fr)
      minmax(390px, 1.2fr) !important;
  }
}

@media (max-width: 900px) {
  .odp-param-row {
    grid-template-columns: 1fr !important;
  }

  .odp-threshold-slider,
  .odp-param-row .block.odp-threshold-slider,
  .odp-param-row .block:has(input[type="range"]) {
    min-width: 0 !important;
  }
}

"""


def _create_user_tabs() -> None:
    with gr.Tabs():
        with gr.TabItem("单图检测"):
            create_single_detection_ui()
        with gr.TabItem("文件夹检测"):
            create_folder_detection_ui()
        with gr.TabItem("视频检测"):
            create_video_detection_ui()
        with gr.TabItem("实时摄像头"):
            create_live_camera_ui()
        with gr.TabItem("模型选择"):
            create_model_selection_ui()
        with gr.TabItem("LLM对话"):
            create_llm_chat_ui()


def _create_admin_tabs() -> None:
    with gr.Tabs():
        with gr.TabItem("Dashboard"):
            create_dashboard_ui()
        with gr.TabItem("单图检测"):
            create_single_detection_ui()
        with gr.TabItem("文件夹检测"):
            create_folder_detection_ui()
        with gr.TabItem("视频检测"):
            create_video_detection_ui()
        with gr.TabItem("实时摄像头"):
            create_live_camera_ui()
        with gr.TabItem("模型选择"):
            create_model_selection_ui()
        with gr.TabItem("模型演示"):
            create_model_demo_ui()
        with gr.TabItem("数据集浏览"):
            create_dataset_browser_ui()
        with gr.TabItem("训练"):
            create_training_ui()
        with gr.TabItem("数据校验"):
            create_validation_ui()
        with gr.TabItem("配置管理"):
            create_config_ui()


_JS_SETUP = """
<style>
.odp-admin-modal {
  position: fixed !important;
  inset: 0 !important;
  z-index: 2147483647 !important;
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  padding: 24px !important;
  border: 0 !important;
  border-radius: 0 !important;
  background: rgba(248, 250, 255, 0.85) !important;
  flex: 0 0 auto !important;
  min-width: 0 !important;
  max-width: none !important;
  min-height: 100vh !important;
  height: 100vh !important;
  width: 100vw !important;
  margin: 0 !important;
  top: 0 !important;
  left: 0 !important;
  right: 0 !important;
  bottom: 0 !important;
}
.odp-admin-modal.odp-modal-hidden { display: none !important; }
</style>
<img alt="" aria-hidden="true" src="/__odp_bootstrap_missing__.gif" style="display:none" onerror="
(function(){
  if(window.__odpBootstrapDone)return;
  window.__odpBootstrapDone=true;
  function resetContentScroll(){
    requestAnimationFrame(function(){
      requestAnimationFrame(function(){
        var root=document.scrollingElement||document.documentElement;
        if(root)root.scrollTop=0;
        document.documentElement.scrollTop=0;
        document.body.scrollTop=0;
        document.querySelectorAll('.tabs,.tabitem').forEach(function(node){node.scrollTop=0;});
      });
    });
  }
  function restoreAdmin(){
    if(localStorage.getItem('odp_admin')!=='1')return;
    var user=document.querySelector('.odp-user-layer');
    var admin=document.querySelector('.odp-admin-layer');
    if(user&&admin){
      user.classList.add('odp-layer-hidden');
      admin.classList.remove('odp-layer-hidden');
      resetContentScroll();
    }else{
      setTimeout(restoreAdmin,80);
    }
  }
  window.odpResetContentScroll=resetContentScroll;
  document.addEventListener('click',function(event){
    var target=event.target;
    var tab=target&&target.closest?target.closest('[role=tab]'):null;
    if(tab&&tab.closest('.odp-user-layer,.odp-admin-layer'))resetContentScroll();
  },true);
  restoreAdmin();
})();
this.onerror=null;this.remove();
">
"""


def create_app() -> gr.Blocks:
    get_logger(
        base_path=LOGGING_DIR,
        log_type="webui",
        log_level=logging.INFO,
        logger_name="odp-webui",
    )
    logger.info("创建低空智瞰 Gradio UI")

    with gr.Blocks(
        title="低空智瞰",
    ) as app:
        gr.HTML(_JS_SETUP)
        with gr.Column(elem_classes=["odp-shell"]):
            gr.HTML(
                """
                <button
                    type="button"
                    class="odp-sidebar-toggle"
                    aria-label="切换边栏"
                    aria-expanded="true"
                    onclick="
                        var shell = document.querySelector('.odp-shell');
                        var collapsed = shell
                            ? shell.classList.toggle('odp-sidebar-collapsed')
                            : document.documentElement.classList.toggle('odp-sidebar-collapsed');
                        this.setAttribute('aria-expanded', String(!collapsed));
                    "
                >☰</button>
                """,
                elem_classes=["odp-sidebar-toggle-wrap"],
            )
            gr.HTML(
                '<section class="odp-title-art" aria-label="低空智瞰 航拍智能目标识别与检测系统" style="cursor:pointer"'
                ' onclick="'
                "var t=document.querySelectorAll('[role=tablist] button[role=tab]');"
                "if(t.length>0)t[0].click();"
                '" title="点击回到首页">'
                '<span class="odp-logo-mark" aria-hidden="true"></span>'
                '<span class="odp-brand-copy">'
                "<strong>低空智瞰</strong>"
                "<small>航拍智能目标识别与检测系统</small>"
                "</span>"
                "</section>",
                elem_classes=["odp-title"],
            )
            with gr.Column(visible=True, elem_classes=["odp-user-layer"]):
                with gr.Row(elem_classes=["odp-mode-bar"]):
                    gr.HTML(
                        """
                        <button type="button" class="odp-gear-button" aria-label="进入管理员模式"
                        onclick="
                            document.querySelector('.odp-admin-modal').classList.remove('odp-modal-hidden');
                            var i = document.querySelector('.odp-admin-password-input');
                            if(i)i.value='';
                            var e = document.querySelector('.odp-admin-error-text');
                            if(e){e.textContent='';e.classList.remove('is-visible');}
                            setTimeout(function(){ if(i)i.focus(); }, 0);
                        ">&#9881;</button>
                        """,
                    )
                with gr.Row(elem_classes=["odp-user-head"]):
                    gr.HTML(
                        """
                        <div class="odp-user-title-card">
                            <div>
                                <span>用户工作台</span>
                                <small>低空智瞰 · 目标检测与结果查看</small>
                            </div>
                        </div>
                        """,
                        elem_classes=["odp-user-title"],
                    )
                _create_user_tabs()

            with gr.Column(visible=True, elem_classes=["odp-admin-layer", "odp-layer-hidden"]):
                with gr.Row(elem_classes=["odp-admin-head"]):
                    gr.HTML(
                        """
                        <div class="odp-admin-title-card">
                            <div>
                                <span>管理员工作台</span>
                                <small>低空智瞰 · 系统配置与模型运维</small>
                            </div>
                        </div>
                        """,
                        elem_classes=["odp-admin-title"],
                    )
                    gr.HTML(
                        """
                        <button class="odp-return-user-btn" style="
                            min-height:52px;padding:0 24px;
                            border-radius:14px;border:1px solid var(--odp-line);
                            background:var(--odp-card);color:var(--odp-text);
                            font-weight:800;font-size:15px;cursor:pointer;
                        "
                         onclick="
                             document.querySelector('.odp-user-layer').classList.remove('odp-layer-hidden');
                             document.querySelector('.odp-admin-layer').classList.add('odp-layer-hidden');
                             document.querySelector('.odp-admin-modal').classList.add('odp-modal-hidden');
                             localStorage.setItem('odp_admin','0');
                             if(window.odpResetContentScroll) window.odpResetContentScroll();
                             var i = document.querySelector('.odp-admin-password-input');
                             if(i)i.value='';
                             var e = document.querySelector('.odp-admin-error-text');
                             if(e){e.textContent='';e.classList.remove('is-visible');}
                         ">返回用户模式</button>
                        """,
                    )
                _create_admin_tabs()

            with gr.Column(elem_classes=["odp-admin-modal", "odp-modal-hidden"]):
                with gr.Group(elem_classes=["odp-modal-card"]):
                    gr.HTML(
                        """
                        <div class="odp-admin-login">
                            <div class="odp-admin-login-title">
                                <strong>管理员模式</strong>
                                <small>输入密码后进入系统配置与模型运维</small>
                            </div>
                            <label class="odp-admin-form-field">
                                <span>管理员密码</span>
                                <input class="odp-admin-password-input" type="password" placeholder="请输入密码"
                                    autocomplete="current-password"
                                    oninput="
                                        var e=document.querySelector('.odp-admin-error-text');
                                        if(e){e.textContent='';e.classList.remove('is-visible');}
                                    "
                                    onkeydown="
                                        if(event.key==='Enter'){
                                            event.preventDefault();
                                            document.querySelector('.odp-admin-confirm-btn').click();
                                        }
                                    "
                                />
                            </label>
                            <div class="odp-admin-error-text" role="status" aria-live="polite"></div>
                            <div class="odp-modal-actions">
                                <button class="odp-admin-cancel-btn" type="button"
                                onclick="
                                    document.querySelector('.odp-admin-modal').classList.add('odp-modal-hidden');
                                    var i = document.querySelector('.odp-admin-password-input');
                                    if(i)i.value='';
                                    var e = document.querySelector('.odp-admin-error-text');
                                    if(e){e.textContent='';e.classList.remove('is-visible');}
                                ">取消</button>
                                <button class="odp-admin-confirm-btn" type="button"
                                onclick="
                                    var i = document.querySelector('.odp-admin-password-input');
                                    var e = document.querySelector('.odp-admin-error-text');
                                    var v = i ? i.value : '';
                                    if(v === '0000') {
                                        document.querySelector('.odp-user-layer').classList.add('odp-layer-hidden');
                                        document.querySelector('.odp-admin-layer').classList.remove('odp-layer-hidden');
                                        document.querySelector('.odp-admin-modal').classList.add('odp-modal-hidden');
                                        localStorage.setItem('odp_admin','1');
                                        if(window.odpResetContentScroll) window.odpResetContentScroll();
                                        if(i)i.value='';
                                        if(e){e.textContent='';e.classList.remove('is-visible');}
                                    } else {
                                        if(e){e.textContent='密码错误';e.classList.add('is-visible');}
                                    }
                                ">进入</button>
                            </div>
                        </div>
                        """,
                    )
    return app


def _ensure_backend_running(timeout: float = 10.0) -> bool:
    """检测后端是否运行，未运行则自动启动。"""
    try:
        urllib.request.urlopen(f"{BACKEND_URL}/health", timeout=1)
        logger.info("后端服务已在运行")
        return True
    except Exception:
        pass

    logger.info("后端未运行，正在自动启动...")
    backend_main = BACKEND_DIR / "main.py"
    if not backend_main.exists():
        logger.warning("找不到后端入口: %s", backend_main)
        return False

    subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(BACKEND_DIR),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
    )

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(f"{BACKEND_URL}/health", timeout=1)
            logger.info("后端服务启动成功")
            return True
        except Exception:
            time.sleep(0.5)

    logger.warning("后端服务启动超时（%.0fs），Dashboard 功能暂不可用", timeout)
    return False


def main() -> None:
    import tempfile
    import threading
    threading.Thread(target=_ensure_backend_running, daemon=True).start()
    print(f"🌐 浏览器访问: http://127.0.0.1:7860")
    create_app().launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=APP_CSS,
        allowed_paths=[str(ASSETS_DIR), tempfile.gettempdir()],
    )


if __name__ == "__main__":
    main()
