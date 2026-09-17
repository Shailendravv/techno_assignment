# Project Changelog - Chat Interface UI

This document traces all the major changes, refactors, and bug fixes implemented during the development of the Chat Interface project.

---

## [2026-05-16] - API Integration & Mocking
### Added
- **TanStack Query Integration**: Installed `@tanstack/react-query` and configured `QueryClientProvider` for robust state management of API calls.
- **Axios Service Layer**: Created `src/services/api.js` to handle multipart/form-data requests (text + Excel files).
- **Mock API System**: 
  - Created `public/mock_api_response.json` to simulate backend responses.
  - Implemented a simulated network delay (1.5s) in the service layer.
  - Added a placeholder Excel file `public/demo_processed.xlsx` for download testing.
- **Dynamic Download Button**: AI responses now dynamically render a download link if the backend (or mock) provides a file URL.
- **Input Feedback**: Added a file badge in the `InputBar` to show the selected `.xlsx` file and a "Clear" button to remove it.
- **Environment Configuration**: Added `.env` file for easy switching between local development and production API URLs.

### Fixed
- **Missing Icon Imports**: Resolved `ReferenceError` for `MdOutlineAttachFile` and added `IoCloudDownloadOutline`, `IoSend`, and `IoReload`.
- **Loading UX**: Added a spinning reload icon and disabled the input during active API processing to prevent duplicate submissions.

---

## [2026-05-16] - Recent Updates

### Added
- **File Upload Capability**: 
  - Integrated a hidden file input triggered by the attachment icon.
  - Added `fileInputRef` and `handleUploadClick` logic.
- **Sticky Header**: 
  - Introduced a floating header with `backdrop-blur-md` and a logo placeholder.
  - Added "AI Assistant" branding and "History" buttons.
- **Bottom Layout Mask**:
  - Implemented a `bg-gradient-to-t` mask using the theme color (`#f7f4f0`) to prevent messages from being visible underneath the floating input bar.

### Changed
- **Iconography**:
  - Replaced the generic `FaPlus` icon with a clean, vertical paperclip icon (`MdOutlineAttachFile`) to match modern chat UI standards.
- **Theming**:
  - Transitioned from pure `bg-white` to a softer cream background (`#f7f4f0`) to improve visual comfort ("disturbing the eyes" fix).
- **Architecture (Refactoring)**:
  - **`Header.jsx`**: Extracted the header into a standalone component.
  - **`InputBar.jsx`**: Extracted the bottom input field, attachment button, and hidden file input into a standalone component.

### Fixed
- **Uncaught ReferenceError**: 
  - Resolved an issue where `MdOutlineAttachFile` was undefined in the newly created `InputBar.jsx` due to missing imports.
- **Prop Synchronization**:
  - Ensured all state handlers (`setMessage`, `handleSend`, etc.) and refs are correctly passed from the main `ChatInterface` to the `InputBar` sub-component.
- **Layout Corruption**:
  - Fixed various structural HTML/JSX errors caused during rapid refactoring.

---

## Project Structure
- `src/components/ChatInterface.jsx`: Main container and message rendering logic.
- `src/components/Header.jsx`: Sticky top navigation and branding.
- `src/components/InputBar.jsx`: Floating input field and attachment controls.
- `src/pages/Home.jsx`: Main page wrapper with theme background (`#f7f4f0`).

---
*Note: This log serves as a reference for future modifications and to ensure consistency across components.*
