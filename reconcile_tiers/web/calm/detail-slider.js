export function bindDetailSlider(input, onChange) {
  input.addEventListener("input", () => onChange(Number(input.value)));
}

