(() => {
  const ids = ["a", "b", "c"];
  const inputs = Object.fromEntries(
    ids.map((id) => [id, document.querySelector(`[data-unknown="${id}"]`)])
  );
  const outputs = Object.fromEntries(
    ["y0", "y1", "y2", "combined"].map((id) => [
      id,
      document.querySelector(`[data-result="${id}"]`),
    ])
  );

  if (Object.values(inputs).some((node) => !node)) return;

  const render = () => {
    const a = Number(inputs.a.checked);
    const b = Number(inputs.b.checked);
    const c = Number(inputs.c.checked);
    const y0 = a ^ b;
    const y1 = a ^ c;
    const y2 = b ^ c;
    const combined = y0 ^ y1 ^ y2;

    outputs.y0.value = String(y0);
    outputs.y1.value = String(y1);
    outputs.y2.value = String(y2);
    outputs.combined.value = String(combined);
    outputs.combined.parentElement.classList.toggle("good", combined === 0);
  };

  ids.forEach((id) => inputs[id].addEventListener("change", render));
  render();
})();
