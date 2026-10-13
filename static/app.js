(function () {
  const $ = (id) => document.getElementById(id);
  const buttons = ["btn-analyze", "btn-summarize", "btn-insights", "btn-ask"].map($);

  function setStatus(message, isError) {
    const el = $("status");
    el.textContent = message || "";
    el.classList.toggle("error", Boolean(isError));
  }

  function showAnalysis(a) {
    if (!a) return;
    $("empty").hidden = true;
    $("readout").hidden = false;
    // Map the compound score (-1..1) onto the gauge (0..100%).
    $("marker").style.left = ((a.score + 1) / 2) * 100 + "%";
    const label = $("sentiment-label");
    label.textContent = a.sentiment;
    label.className = a.sentiment;
    $("sentiment-score").textContent = "score " + (a.score >= 0 ? "+" : "") + a.score.toFixed(2);
    const via = { rules: "keyword rules", model: "model, " + Math.round(a.model_confidence * 100) + "% confidence",
                  default: "no rule matched and the model was not confident" };
    $("category").textContent = a.category + " (" + via[a.category_source] + ")";
    const names = [].concat(a.entities.companies, a.entities.indices, a.entities.institutions);
    $("entities").textContent = names.length ? names.join(", ") : "No known companies, indices or institutions";
  }

  function showLLM(title, text) {
    $("llm-block").hidden = false;
    $("llm-title").textContent = title;
    $("llm-output").textContent = text || "(The model returned no text. Try again.)";
  }

  async function call(path, payload, title) {
    buttons.forEach((b) => (b.disabled = true));
    setStatus("Working. The first language-model call can take a minute while the model loads.");
    try {
      const resp = await fetch(path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await resp.json();
      if (data.analysis) showAnalysis(data.analysis);
      if (!resp.ok) { setStatus(data.error || "Something went wrong.", true); return; }
      if (title) showLLM(title, data.result);
      setStatus("");
    } catch (err) {
      setStatus("Could not reach the server. Is app.py still running?", true);
    } finally {
      buttons.forEach((b) => (b.disabled = false));
    }
  }

  const text = () => $("article").value;

  $("btn-analyze").addEventListener("click", () => call("/api/analyze", { text: text() }, null));
  $("btn-summarize").addEventListener("click", () => call("/api/summarize", { text: text() }, "Summary"));
  $("btn-insights").addEventListener("click", () => call("/api/insights", { text: text() }, "Key takeaways"));
  $("btn-ask").addEventListener("click", () =>
    call("/api/ask", { text: text(), question: $("question").value }, "Answer"));
  $("question").addEventListener("keydown", (e) => { if (e.key === "Enter") $("btn-ask").click(); });

  $("sample").addEventListener("change", (e) => {
    if (e.target.value !== "") $("article").value = window.CURATED[Number(e.target.value)].text;
  });

  $("file").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => { $("article").value = String(reader.result); $("sample").value = ""; };
    reader.readAsText(file);
  });
})();
