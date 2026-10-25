# UBIPR Both-Eyes Dataset Audit Report

## 1. Executive Summary
- **Source Directory:** `c:\Users\Piyush\OneDrive\Desktop\ubipr2\both_eyes`
- **Total Files Inventoried:** 11020
- **Ignored Files / OS Artifacts:** 2 (.DS_Store, ._.DS_Store)
- **Valid Paired Samples:** 5509 (100% paired .jpg images and .png masks)
- **Unique Subjects:** 261
- **Unreadable / Corrupt Images:** 0
- **Unparseable Filenames:** 0

## 2. Naming Convention & Label Parsing Verification
Filenames follow the official UBIPR naming standard:
`C<subject_id>_S<session_id>_I<image_id>.jpg` (and matching `.png`)
- **Subject ID:** Formatted as `C1` through `C...` (e.g. `C101`)
- **Session:** Session `S1` or `S2`
- **Image ID:** Image index `I1`, `I2`, ...

### Session Breakdown
| Session | Image Count | Percentage |
|:---:|:---:|:---:|
| Session 1 | 3914 | 71.05% |
| Session 2 | 1595 | 28.95% |

### Subject Sample Statistics
- **Minimum images per subject:** 14
- **Maximum images per subject:** 33
- **Mean images per subject:** 21.11
- **Median images per subject:** 15

> **Protocol Compatibility:** Because every subject contains at least 14 samples (>= 10), enrollment sizes **1, 3, 5, and 10** are fully supported across all subjects.

## 3. Image Properties & Mask Semantics
- **Channels:** [3] (3-channel BGR color)
- **Data Type:** ['uint8']
- **Sample Min Resolution:** 1067x368 px
- **Sample Max Resolution:** 2781x959 px
- **Mask Classes Discovered:** [0, 85, 170, 255]
  - `0`: Background / Facial Skin
  - `85`: Iris Region (Both Left & Right Eyes)
  - `170`: Sclera Region (Lateral & Medial sides of both eyes)
  - `255`: Eyebrows

## 4. Visual Inspection Contact Sheet
A 16-sample labeled contact sheet with green iris mask overlays was generated and saved to:
`reports/contact_sheet.jpg` and `outputs/figures/contact_sheet.png`.

## 5. Audit Conclusion
The dataset conforms precisely to the UBIPR both-eyes benchmark specifications with zero missing masks, zero unreadable images, and zero unparseable labels.
