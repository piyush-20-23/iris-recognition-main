document.addEventListener('DOMContentLoaded', () => {
  // Elements
  const systemStatusBadge = document.getElementById('systemStatusBadge');
  const systemStatusText = document.getElementById('systemStatusText');
  const enrolledSubjectsList = document.getElementById('enrolledSubjectsList');
  const statEnrolled = document.getElementById('statEnrolled');

  const scannerForm = document.getElementById('scannerForm');
  const modeIdentify = document.getElementById('modeIdentify');
  const modeVerify = document.getElementById('modeVerify');
  const methodSelect = document.getElementById('methodSelect');
  const claimedSubjectGroup = document.getElementById('claimedSubjectGroup');
  const claimedSubjectInput = document.getElementById('claimedSubjectInput');

  const imageFileInput = document.getElementById('imageFileInput');
  const imageDropZone = document.getElementById('imageDropZone');
  const dropPrompt = document.getElementById('dropPrompt');
  const imagePreview = document.getElementById('imagePreview');
  const maskFileInput = document.getElementById('maskFileInput');

  const submitBtn = document.getElementById('submitBtn');
  const btnSpinner = document.getElementById('btnSpinner');
  const errorAlert = document.getElementById('errorAlert');

  const resultsPlaceholder = document.getElementById('resultsPlaceholder');
  const resultsContent = document.getElementById('resultsContent');
  const badgeQuality = document.getElementById('badgeQuality');
  const valQualityText = document.getElementById('valQualityText');
  const valSegText = document.getElementById('valSegText');
  const valGalleryText = document.getElementById('valGalleryText');

  const verificationResultCard = document.getElementById('verificationResultCard');
  const identificationResultsContainer = document.getElementById('identificationResultsContainer');

  let selectedImageFile = null;

  // 1. Initial Health & Metadata Check
  async function checkHealth() {
    try {
      const res = await fetch('/api/v1/health');
      if (!res.ok) throw new Error('API offline');
      const data = await res.json();
      systemStatusText.textContent = `Online (${data.enrolled_subjects_count} Subjects Enrolled)`;
      statEnrolled.textContent = `${data.enrolled_subjects_count} Subjects`;
      valGalleryText.textContent = `${data.enrolled_subjects_count} Subjects`;

      // Populate datalist for claimed subject verification
      if (data.enrolled_subjects) {
        enrolledSubjectsList.innerHTML = '';
        data.enrolled_subjects.forEach(subj => {
          const opt = document.createElement('option');
          opt.value = subj;
          enrolledSubjectsList.appendChild(opt);
        });
        if (data.enrolled_subjects.length > 0) {
          claimedSubjectInput.placeholder = `e.g. ${data.enrolled_subjects[0]}`;
        }
      }
    } catch (e) {
      systemStatusText.textContent = 'Service Offline';
      systemStatusBadge.style.backgroundColor = 'rgba(239, 68, 68, 0.15)';
      systemStatusBadge.style.color = '#f87171';
    }
  }

  checkHealth();

  // 2. Mode Toggle (Identify vs Verify)
  function handleModeChange() {
    if (modeVerify.checked) {
      claimedSubjectGroup.style.display = 'block';
      if (methodSelect.value === 'all') {
        methodSelect.value = 'gabor_hamming';
      }
      submitBtn.querySelector('.btn-text').textContent = 'Verify Claimed Identity';
    } else {
      claimedSubjectGroup.style.display = 'none';
      submitBtn.querySelector('.btn-text').textContent = 'Execute Biometric Scan';
    }
  }

  modeIdentify.addEventListener('change', handleModeChange);
  modeVerify.addEventListener('change', handleModeChange);

  // 3. Image Upload & Drag-and-Drop Handling
  imageDropZone.addEventListener('click', () => imageFileInput.click());

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
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleImageSelection(e.dataTransfer.files[0]);
    }
  });

  imageFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleImageSelection(e.target.files[0]);
    }
  });

  function handleImageSelection(file) {
    if (!file.type.startsWith('image/')) {
      showError('Please select a valid image file (*.jpg, *.png).');
      return;
    }
    selectedImageFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      imagePreview.src = e.target.result;
      imagePreview.style.display = 'block';
      dropPrompt.style.display = 'none';
      submitBtn.disabled = false;
      clearError();
    };
    reader.readAsDataURL(file);
  }

  function showError(msg) {
    errorAlert.textContent = msg;
    errorAlert.style.display = 'block';
  }

  function clearError() {
    errorAlert.textContent = '';
    errorAlert.style.display = 'none';
  }

  // 4. Form Submission
  scannerForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!selectedImageFile) {
      showError('Please upload an eye image.');
      return;
    }

    const isVerify = modeVerify.checked;
    const claimedSubject = claimedSubjectInput.value.trim();

    if (isVerify && !claimedSubject) {
      showError('Please specify the Claimed Subject ID to verify.');
      return;
    }

    clearError();
    submitBtn.disabled = true;
    btnSpinner.style.display = 'inline-block';

    const formData = new FormData();
    formData.append('image', selectedImageFile);

    if (maskFileInput.files && maskFileInput.files.length > 0) {
      formData.append('mask', maskFileInput.files[0]);
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
        throw new Error(data.message || 'Error occurred during scan.');
      }

      displayResults(data, isVerify);
    } catch (err) {
      showError(err.message || 'Network error occurred.');
    } finally {
      submitBtn.disabled = false;
      btnSpinner.style.display = 'none';
    }
  });

  // 5. Render Scan Results
  function displayResults(data, isVerify) {
    resultsPlaceholder.style.display = 'none';
    resultsContent.style.display = 'block';

    // Update Quality Ribbon
    if (data.quality) {
      valQualityText.textContent = data.quality.is_usable ?
        `Usable (Sharpness: ${data.quality.sharpness})` :
        `Warning: ${data.quality.issues.join(', ')}`;
    }

    // Update Segmentation Status
    if (data.segmentation) {
      valSegText.textContent = `${data.segmentation.detected_eyes_count} Eye(s) (${data.segmentation.method})`;
    }

    if (isVerify) {
      // Show Verification View
      identificationResultsContainer.style.display = 'none';
      verificationResultCard.style.display = 'block';

      const isMatch = data.is_verified_match;
      const header = document.getElementById('verDecisionHeader');
      const icon = document.getElementById('verDecisionIcon');
      const title = document.getElementById('verDecisionTitle');
      const sub = document.getElementById('verDecisionSub');

      if (isMatch) {
        header.className = 'ver-decision-header match';
        icon.textContent = '✅';
        title.textContent = 'IDENTITY VERIFIED (MATCH)';
        sub.textContent = `Claimed identity matches within validation operating threshold.`;
      } else {
        header.className = 'ver-decision-header no-match';
        icon.textContent = '❌';
        title.textContent = 'VERIFICATION FAILED (NO MATCH)';
        sub.textContent = `Claimed identity rejected against validation operating threshold.`;
      }

      document.getElementById('verClaimedSubj').textContent = data.claimed_subject_id;
      document.getElementById('verRawScore').textContent = `${data.raw_score} (${data.score_type})`;
      document.getElementById('verThreshold').textContent = data.validation_threshold ? data.validation_threshold.toFixed(4) : 'N/A';
      document.getElementById('verSimilarity').textContent = `${(data.normalized_similarity * 100).toFixed(1)}%`;

      const dirNote = data.higher_is_better ?
        'Score Direction: Higher score is better. Match requires score ≥ threshold.' :
        'Score Direction: Lower distance is better. Match requires distance ≤ threshold.';
      document.getElementById('verDirectionNote').textContent = dirNote;

    } else {
      // Show Identification View
      verificationResultCard.style.display = 'none';
      identificationResultsContainer.style.display = 'block';
      identificationResultsContainer.innerHTML = '';

      const methodNames = Object.keys(data.results || {});
      const methodLabels = {
        gabor_hamming: 'Gabor Filter Bank + Hamming Distance',
        lbp_knn: 'LBP Spatial Block Histograms + KNN',
        gabor_svm: 'Gabor Energy Features + Linear SVM'
      };

      const grid = document.createElement('div');
      grid.className = 'method-results-grid';

      methodNames.forEach(mKey => {
        const mData = data.results[mKey];
        const card = document.createElement('div');
        card.className = 'method-card';

        const directionLabel = mData.higher_is_better ? 'Higher Score is Better' : 'Lower Distance is Better';

        let candidatesHtml = '';
        (mData.top_candidates || []).forEach(cand => {
          const simPct = (cand.normalized_similarity * 100).toFixed(1);
          const isTop = cand.rank === 1 ? 'top-rank' : '';
          candidatesHtml += `
            <tr class="${isTop}">
              <td>#${cand.rank}</td>
              <td><strong>${cand.subject_id}</strong></td>
              <td>${cand.raw_score}</td>
              <td>
                <div>${simPct}%</div>
                <div class="sim-bar-container">
                  <div class="sim-bar" style="width: ${simPct}%;"></div>
                </div>
              </td>
            </tr>
          `;
        });

        card.innerHTML = `
          <div class="method-card-header">
            <span class="method-title">${methodLabels[mKey] || mKey}</span>
            <span class="score-direction-pill">${directionLabel}</span>
          </div>
          <table class="results-table">
            <thead>
              <tr>
                <th>Rank</th>
                <th>Subject ID</th>
                <th>Raw Score</th>
                <th>Similarity</th>
              </tr>
            </thead>
            <tbody>
              ${candidatesHtml}
            </tbody>
          </table>
        `;
        grid.appendChild(card);
      });

      identificationResultsContainer.appendChild(grid);
    }
  }
});
