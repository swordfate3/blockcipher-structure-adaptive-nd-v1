(() => {
  const rows = {
    y0: [1, 1, 0],
    y1: [1, 0, 1],
    y2: [0, 1, 1],
  };
  const checks = Object.fromEntries(
    Object.keys(rows).map((id) => [id, document.querySelector(`[data-matrix-row="${id}"]`)])
  );
  const outputs = ["a", "b", "c"].map((id) =>
    document.querySelector(`[data-column-parity="${id}"]`)
  );
  const status = document.querySelector("[data-matrix-status]");

  if (Object.values(checks).some((node) => !node) || outputs.some((node) => !node) || !status) {
    return;
  }

  const render = () => {
    const selected = Object.entries(checks)
      .filter(([, checkbox]) => checkbox.checked)
      .map(([id]) => id);
    const parity = [0, 0, 0];

    selected.forEach((id) => {
      rows[id].forEach((value, column) => {
        parity[column] ^= value;
      });
    });

    parity.forEach((value, index) => {
      outputs[index].value = String(value);
    });

    const proved = selected.length > 0 && parity.every((value) => value === 0);
    status.classList.toggle("good", proved);
    if (selected.length === 0) {
      status.textContent = "还没有选择输出公式。空组合没有密码分析意义。";
    } else if (proved) {
      status.textContent = `成功：${selected.join(" XOR ")} 让 A、B、C 每列都出现偶数次，未决项全部抵消。`;
    } else {
      const remaining = ["A", "B", "C"].filter((_, index) => parity[index] === 1);
      status.textContent = `还不能证明：未决项 ${remaining.join("、")} 仍出现奇数次。`;
    }
  };

  Object.values(checks).forEach((checkbox) => checkbox.addEventListener("change", render));
  render();
})();
