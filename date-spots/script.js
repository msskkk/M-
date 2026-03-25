// =============================================
//  script.js - デートスポットカード生成 & フィルター
//  担当: ミサキ
//  前提: spots-data.js で SPOTS_DATA が定義済み
// =============================================

(function () {
  'use strict';

  // ---------- DOM refs ----------
  const container = document.getElementById('spots-container');
  const categoryBtns = document.querySelectorAll('#category-filter .filter-btn');
  const areaBtns = document.querySelectorAll('#area-filter .filter-btn');
  const budgetSlider = document.getElementById('budget-slider');
  const budgetDisplay = document.getElementById('budget-value') || document.getElementById('budget-display');

  // ---------- カード生成 ----------
  function createSpotCard(spot) {
    const card = document.createElement('article');
    card.className = 'spot-card';
    card.dataset.category = spot.category;
    card.dataset.area = spot.area;
    card.dataset.budget = spot.budget;

    const maxStars = 5;
    const filled = '★'.repeat(spot.rating);
    const empty = '☆'.repeat(maxStars - spot.rating);

    // カテゴリ → タグクラス名マッピング
    const tagClassMap = {
      'カフェ': 'tag-cafe',
      '公園': 'tag-park',
      '水族館': 'tag-aquarium',
      '美術館': 'tag-museum',
      '夜景': 'tag-nightview',
    };
    const tagClass = tagClassMap[spot.category] || '';

    card.innerHTML = `
      <div class="card-emoji">${spot.emoji}</div>
      <div class="card-body">
        <h3>${spot.name}</h3>
        <p class="area">${spot.area}</p>
        <div class="tags">
          <span class="tag ${tagClass}">${spot.category}</span>
        </div>
        <p>${spot.comment}</p>
        <div class="card-footer">
          <span class="price">¥${spot.budget.toLocaleString()}</span>
          <div class="stars" data-stars="${spot.rating}">${filled}${empty}</div>
        </div>
      </div>
    `;

    return card;
  }

  // ---------- フィルター状態取得 ----------
  function getActiveValues(buttons, key) {
    const active = [];
    buttons.forEach(btn => {
      if (btn.classList.contains('active')) {
        active.push(btn.dataset[key]);
      }
    });
    return active;
  }

  // ---------- 予算表示更新 ----------
  function updateBudgetDisplay(value) {
    if (!budgetDisplay) return;
    budgetDisplay.textContent = `〜¥${Number(value).toLocaleString()}`;
  }

  // ---------- フィルター適用 & 描画 ----------
  function applyFilters() {
    const selectedCategories = getActiveValues(categoryBtns, 'category');
    const selectedAreas = getActiveValues(areaBtns, 'area');
    const maxBudget = Number(budgetSlider.value);

    const filtered = SPOTS_DATA.filter(spot => {
      const catOk = selectedCategories.length === 0 || selectedCategories.includes(spot.category);
      const areaOk = selectedAreas.length === 0 || selectedAreas.includes(spot.area);
      const budgetOk = spot.budget <= maxBudget;
      return catOk && areaOk && budgetOk;
    });

    renderCards(filtered);
  }

  // ---------- カード描画（staggered fadeIn） ----------
  function renderCards(spots) {
    container.innerHTML = '';

    if (spots.length === 0) {
      const msg = document.createElement('p');
      msg.className = 'no-results';
      msg.textContent = '条件に合うスポットが見つかりません\uD83E\uDD72';
      container.appendChild(msg);
      return;
    }

    spots.forEach((spot, i) => {
      const card = createSpotCard(spot);
      card.style.opacity = '0';
      card.style.animation = `fadeIn 0.4s ease ${i * 0.05}s forwards`;
      container.appendChild(card);
    });
  }

  // ---------- イベント登録 ----------
  function setupFilterButtons(buttons) {
    buttons.forEach(btn => {
      btn.addEventListener('click', () => {
        btn.classList.toggle('active');
        applyFilters();
      });
    });
  }

  // ---------- 初期化 ----------
  document.addEventListener('DOMContentLoaded', () => {
    // fadeIn keyframes を動的注入（CSS側に無くても動くように）
    if (!document.getElementById('fadeIn-keyframes')) {
      const style = document.createElement('style');
      style.id = 'fadeIn-keyframes';
      style.textContent = `
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(12px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `;
      document.head.appendChild(style);
    }

    setupFilterButtons(categoryBtns);
    setupFilterButtons(areaBtns);

    budgetSlider.addEventListener('input', () => {
      updateBudgetDisplay(budgetSlider.value);
      applyFilters();
    });

    // 初期表示
    updateBudgetDisplay(budgetSlider.value);
    applyFilters();
  });
})();
