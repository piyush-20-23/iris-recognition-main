document.addEventListener('DOMContentLoaded', () => {
  // Navigation & Status Elements
  const systemStatusBadge = document.getElementById('systemStatusBadge');
  const systemStatusText = document.getElementById('systemStatusText');
  const enrolledSubjectsList = document.getElementById('enrolledSubjectsList');
  const resetWorkstationBtn = document.getElementById('resetWorkstationBtn');

  // Form & Mode Controls
  const scannerForm = document.getElementById('scannerForm');
  const modeIdentify = document.getElementById('modeIdentify');
  const modeVerify = document.getElementById('modeVerify');
  const methodSelect = document.getElementById('methodSelect');
  const claimedSubjectGroup = document.getElementById('claimedSubjectGroup');
  const claimedSubjectInput = document.getElementById('claimedSubjectInput');

  // Image Upload Controls
  const imageFileInput = document.getElementById('imageFileInput');
  const imageDropZone = document.getElementById('imageDropZone');
  const dropPrompt = document.getElementById('dropPrompt');
  const imagePreview = document.getElementById('imagePreview');
  const fileInfoBadge = document.getElementById('fileInfoBadge');
  const fileInfoName = document.getElementById('fileInfoName');
  const fileInfoSize = document.getElementById('fileInfoSize');
  const fileInfoClearBtn = document.getElementById('fileInfoClearBtn');

  // Mask Controls
  const maskFileInput = document.getElementById('maskFileInput');
  const maskFeedback = document.getElementById('maskFeedback');
  const maskFeedbackName = document.getElementById('maskFeedbackName');

  // Quick Samples
  const quickSamplesRow = document.getElementById('quickSamplesRow');
  const quickSamplesList = document.getElementById('quickSamplesList');

  // Submission & Feedback
  const submitBtn = document.getElementById('submitBtn');
  const btnSpinner = document.getElementById('btnSpinner');
  const errorAlert = document.getElementById('errorAlert');

  // Results Elements
  const resultsPlaceholder = document.getElementById('resultsPlaceholder');
  const resultsContent = document.getElementById('resultsContent');
  const valQualityText = document.getElementById('valQualityText');
  const valSegText = document.getElementById('valSegText');

  // Verification Display
  const verificationResultCard = document.getElementById('verificationResultCard');
  const verDecisionHeader = document.getElementById('verDecisionHeader');
  const verDecisionIconContainer = document.getElementById('verDecisionIconContainer');
  const verDecisionTitle = document.getElementById('verDecisionTitle');
  const verDecisionSub = document.getElementById('verDecisionSub');
  const verClaimedSubj = document.getElementById('verClaimedSubj');
  const verRawScore = document.getElementById('verRawScore');
  const verThreshold = document.getElementById('verThreshold');
  const verSimilarity = document.getElementById('verSimilarity');
  const verDirectionNote = document.getElementById('verDirectionNote');

  // Identification Display
  const identificationResultsContainer = document.getElementById('identificationResultsContainer');

  let selectedImageFile = null;

  // 1. Initial Health Check
  async function checkHealth() {
    try {
      const res = await fetch('/api/v1/health');
      if (!res.ok) throw new Error('API offline');
      const data = await res.json();
      if (systemStatusText) {
        systemStatusText.textContent = data.is_gallery_enrolled ? 'System Ready' : 'Standby Mode';
      }
      if (systemStatusBadge) {
        systemStatusBadge.classList.remove('offline');
      }

      // Populate datalist for claimed subject verification
      if (enrolledSubjectsList && data.enrolled_subjects && data.enrolled_subjects.length > 0) {
        enrolledSubjectsList.innerHTML = '';
        data.enrolled_subjects.forEach(subj => {
          const opt = document.createElement('option');
          opt.value = subj;
          enrolledSubjectsList.appendChild(opt);
        });
        if (claimedSubjectInput && !claimedSubjectInput.value) {
          claimedSubjectInput.placeholder = `e.g. ${data.enrolled_subjects[0]}`;
        }
      }
    } catch (e) {
      if (systemStatusText) {
        systemStatusText.textContent = 'Service Offline';
      }
      if (systemStatusBadge) {
        systemStatusBadge.classList.add('offline');
      }
    }
  }

  // 2. Load Quick Test Samples
  async function loadQuickSamples() {
    try {
      const res = await fetch('/api/v1/samples');
      if (!res.ok) return;
      const data = await res.json();
      if (data.samples && data.samples.length > 0 && quickSamplesList) {
        quickSamplesList.innerHTML = '';
        data.samples.forEach(sample => {
          const btn = document.createElement('button');
          btn.type = 'button';
          btn.className = 'sample-btn';
          btn.textContent = `${sample.subject_id}`;
          btn.title = `Load sample probe ${sample.filename}`;
          btn.addEventListener('click', (e) => {
            e.stopPropagation();
            loadSampleFile(sample.url, sample.filename, sample.subject_id);
          });
          quickSamplesList.appendChild(btn);
        });
        if (quickSamplesRow) {
          quickSamplesRow.style.display = 'block';
        }
      }
    } catch (e) {
      console.warn('Could not load sample images:', e);
    }
  }

  async function loadSampleFile(url, filename, subjectId) {
    try {
      clearError();
      const res = await fetch(url);
      if (!res.ok) throw new Error('Could not fetch sample probe image');
      const blob = await res.blob();
      const file = new File([blob], filename, { type: 'image/jpeg' });
      handleImageSelection(file);

      // Pre-fill claimed subject in verification mode
      if (modeVerify.checked && claimedSubjectInput) {
        claimedSubjectInput.value = subjectId;
      }
    } catch (e) {
      showError(`Error loading sample: ${e.message}`);
    }
  }

  checkHealth();
  loadQuickSamples();

  // 3. Mode Toggle (Identify vs Verify)
  function handleModeChange() {
    if (modeVerify.checked) {
      if (claimedSubjectGroup) claimedSubjectGroup.style.display = 'block';
      if (methodSelect && methodSelect.value === 'all') {
        methodSelect.value = 'ensemble_fusion';
      }
      submitBtn.querySelector('.btn-text').textContent = 'Verify Claimed Identity';

      // Auto pre-fill claimed subject from current file if available
      if (selectedImageFile && claimedSubjectInput) {
        const match = selectedImageFile.name.match(/^C(\d+)/i);
        if (match && !claimedSubjectInput.value) {
          claimedSubjectInput.value = `C${match[1]}`;
        }
      }
    } else {
      if (claimedSubjectGroup) claimedSubjectGroup.style.display = 'none';
      submitBtn.querySelector('.btn-text').textContent = 'Execute Biometric Scan';
    }
  }

  if (modeIdentify) modeIdentify.addEventListener('change', handleModeChange);
  if (modeVerify) modeVerify.addEventListener('change', handleModeChange);

  // 4. Image Upload & Drag-and-Drop
  if (imageDropZone) {
    imageDropZone.addEventListener('click', () => {
      imageFileInput.click();
    });

    imageDropZone.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') {
        e.preventDefault();
        imageFileInput.click();
      }
    });

    ['dragenter', 'dragover'].forEach(evt => {
      imageDropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        imageDropZone.classList.add('dragover');
      });
    });

    ['dragleave', 'drop'].forEach(evt => {
      imageDropZone.addEventListener(evt, (e) => {
        e.preventDefault();
        imageDropZone.classList.remove('dragover');
      });
    });

    imageDropZone.addEventListener('drop', (e) => {
      const files = e.dataTransfer.files;
      if (files && files.length > 0) {
        handleImageSelection(files[0]);
      }
    });
  }

  if (imageFileInput) {
    imageFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleImageSelection(e.target.files[0]);
      }
    });
  }

  // Mask file input feedback
  if (maskFileInput) {
    maskFileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files.length > 0) {
        maskFeedbackName.textContent = e.target.files[0].name;
        maskFeedback.style.display = 'inline-flex';
      } else {
        maskFeedback.style.display = 'none';
      }
    });
  }

  function handleImageSelection(file) {
    if (!file.type.match('image.*') && !file.name.match(/\.(jpg|jpeg|png|bmp)$/i)) {
      showError('Please upload a valid image file (*.jpg, *.png, *.bmp).');
      return;
    }
    clearError();
    selectedImageFile = file;

    // Display filename and size
    fileInfoName.textContent = file.name;
    const sizeKb = Math.round(file.size / 1024);
    fileInfoSize.textContent = `(${sizeKb} KB)`;
    fileInfoBadge.style.display = 'flex';

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      imagePreview.style.display = 'block';
      dropPrompt.style.display = 'none';
    };
    reader.readAsDataURL(file);

    // Auto-detect subject if in verification mode
    if (modeVerify.checked && claimedSubjectInput) {
      const match = file.name.match(/^C(\d+)/i);
      if (match) {
        claimedSubjectInput.value = `C${match[1]}`;
      }
    }

    submitBtn.disabled = false;
  }

  function resetImageSelection() {
    selectedImageFile = null;
    if (imageFileInput) imageFileInput.value = '';
    if (imagePreview) {
      imagePreview.src = '';
      imagePreview.style.display = 'none';
    }
    if (dropPrompt) dropPrompt.style.display = 'flex';
    if (fileInfoBadge) fileInfoBadge.style.display = 'none';
  }

  if (fileInfoClearBtn) {
    fileInfoClearBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      resetImageSelection();
      clearError();
    });
  }

  // Reset Workstation Button
  if (resetWorkstationBtn) {
    resetWorkstationBtn.addEventListener('click', () => {
      resetImageSelection();
      if (maskFileInput) maskFileInput.value = '';
      if (maskFeedback) maskFeedback.style.display = 'none';
      if (claimedSubjectInput) claimedSubjectInput.value = '';
      clearError();
      if (resultsPlaceholder) resultsPlaceholder.style.display = 'flex';
      if (resultsContent) resultsContent.style.display = 'none';
    });
  }

  function showError(msg) {
    if (errorAlert) {
      errorAlert.textContent = msg;
      errorAlert.style.display = 'block';
    }
  }

  function clearError() {
    if (errorAlert) {
      errorAlert.textContent = '';
      errorAlert.style.display = 'none';
    }
  }

  // 5. Submit Form
  if (scannerForm) {
    scannerForm.addEventListener('submit', async (e) => {
      e.preventDefault();

      if (!selectedImageFile) {
        showError('Please select or drop a probe image first.');
        submitBtn.disabled = false;
        return;
      }

      const isVerify = modeVerify.checked;
      const claimedSubject = claimedSubjectInput ? claimedSubjectInput.value.trim() : '';

      if (isVerify && !claimedSubject) {
        showError('Please specify the Claimed Subject ID to verify.');
        return;
      }

      clearError();
      submitBtn.disabled = true;
      if (btnSpinner) btnSpinner.style.display = 'inline-block';

      const formData = new FormData();
      formData.append('image', selectedImageFile, selectedImageFile.name);

      if (maskFileInput && maskFileInput.files && maskFileInput.files.length > 0) {
        formData.append('mask', maskFileInput.files[0], maskFileInput.files[0].name);
      }

      formData.append('method', methodSelect.value);

      const endpoint = isVerify ? '/api/v1/verify' : '/api/v1/identify';
      if (isVerify) {
        formData.append('claimed_subject_id', claimedSubject);
      }

      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          body: formData
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.message || 'Error occurred during biometric scan.');
        }

        displayResults(data, isVerify);
      } catch (err) {
        showError(err.message || 'A network error occurred while communicating with the biometric service.');
      } finally {
        submitBtn.disabled = false;
        if (btnSpinner) btnSpinner.style.display = 'none';
      }
    });
  }

  // 6. Render Biometric Results
  function displayResults(data, isVerify) {
    if (resultsPlaceholder) resultsPlaceholder.style.display = 'none';
    if (resultsContent) resultsContent.style.display = 'flex';

    // Diagnostics tags
    if (data.quality && valQualityText) {
      const sharpnessStr = typeof data.quality.sharpness === 'number' ?
        data.quality.sharpness.toFixed(1) : data.quality.sharpness;
      valQualityText.textContent = data.quality.is_usable ?
        `Valid (Sharpness: ${sharpnessStr})` :
        `Quality Warning: ${data.quality.issues.join(', ')}`;
    }

    if (data.segmentation && valSegText) {
      const segMethodName = data.segmentation.method === 'ground_truth_mask' ?
        'Ground Truth Mask' : 'Automated Detector';
      valSegText.textContent = `${data.segmentation.detected_eyes_count} Eyes Located (${segMethodName})`;
    }

    if (isVerify) {
      // 1:1 Verification Results
      if (identificationResultsContainer) identificationResultsContainer.style.display = 'none';
      if (verificationResultCard) verificationResultCard.style.display = 'block';

      const isMatch = data.is_verified_match;

      if (isMatch) {
        verDecisionHeader.className = 'ver-header match';
        verDecisionIconContainer.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="ver-svg" aria-hidden="true">
            <polyline points="20 6 9 17 4 12"/>
          </svg>
        `;
        verDecisionTitle.textContent = 'IDENTITY VERIFIED (MATCH)';
        verDecisionSub.textContent = 'Probe matches claimed identity within validation operating threshold.';
      } else {
        verDecisionHeader.className = 'ver-header reject';
        verDecisionIconContainer.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" class="ver-svg" aria-hidden="true">
            <circle cx="12" cy="12" r="10"/>
            <line x1="15" y1="9" x2="9" y2="15"/>
            <line x1="9" y1="9" x2="15" y2="15"/>
          </svg>
        `;
        verDecisionTitle.textContent = 'VERIFICATION REJECTED (NO MATCH)';
        verDecisionSub.textContent = 'Probe distance exceeds allowable operating threshold for claimed identity.';
      }

      if (verClaimedSubj) verClaimedSubj.textContent = data.claimed_subject_id;
      if (verRawScore) verRawScore.textContent = `${data.raw_score} (${data.score_type})`;
      if (verThreshold) verThreshold.textContent = data.validation_threshold ? data.validation_threshold.toFixed(4) : 'N/A';
      if (verSimilarity) verSimilarity.textContent = `${(data.normalized_similarity * 100).toFixed(1)}%`;

      if (verDirectionNote) {
        const dirNote = data.higher_is_better ?
          'Operating Rule: Higher score indicates identity alignment. Match requires score ≥ threshold.' :
          'Operating Rule: Lower distance indicates identity alignment. Match requires distance ≤ threshold.';
        verDirectionNote.textContent = dirNote;
      }

    } else {
      // 1:N Identification Results
      if (verificationResultCard) verificationResultCard.style.display = 'none';
      if (identificationResultsContainer) {
        identificationResultsContainer.style.display = 'flex';
        identificationResultsContainer.innerHTML = '';

        const methodNames = Object.keys(data.results || {});
        const methodLabels = {
          ensemble_fusion: 'Multi-Modal Ensemble Fusion (Gabor + LBP)',
          gabor_svm: 'Gabor Energy Descriptors + Support Vector Machine',
          lbp_knn: 'Local Binary Patterns (LBP) + K-Nearest Neighbors',
          gabor_hamming: 'Gabor Filter Bank + Masked Hamming Distance'
        };

        methodNames.forEach(mKey => {
          const mData = data.results[mKey];
          const card = document.createElement('div');
          card.className = 'method-item';

          const directionLabel = mData.higher_is_better ? 'Higher Score is Better' : 'Lower Distance is Better';

          let candidatesHtml = '';
          (mData.top_candidates || []).forEach(cand => {
            const simPct = (cand.normalized_similarity * 100).toFixed(1);
            const isTop = cand.rank === 1 ? 'top-candidate' : '';
            candidatesHtml += `
              <tr class="${isTop}">
                <td class="rank-badge">#${cand.rank}</td>
                <td><span class="subject-badge">${cand.subject_id}</span></td>
                <td class="font-mono text-muted">${cand.raw_score}</td>
                <td>
                  <div class="sim-bar-wrap">
                    <span class="font-mono font-medium">${simPct}%</span>
                    <div class="sim-meter" role="progressbar" aria-valuenow="${simPct}" aria-valuemin="0" aria-valuemax="100">
                      <div class="sim-fill" style="width: ${simPct}%;"></div>
                    </div>
                  </div>
                </td>
              </tr>
            `;
          });

          card.innerHTML = `
            <div class="method-header">
              <span class="method-label">${methodLabels[mKey] || mKey}</span>
              <span class="direction-badge">${directionLabel}</span>
            </div>
            <div class="table-wrap">
              <table class="candidate-table">
                <thead>
                  <tr>
                    <th style="width: 55px;">Rank</th>
                    <th>Subject ID</th>
                    <th>Raw Score</th>
                    <th style="width: 150px;">Similarity</th>
                  </tr>
                </thead>
                <tbody>
                  ${candidatesHtml}
                </tbody>
              </table>
            </div>
          `;
          identificationResultsContainer.appendChild(card);
        });
      }
    }
  }
});
