(function () {
  "use strict";

  const form = document.getElementById("search-form");
  const queryInput = document.getElementById("query");
  const sectorSelect = document.getElementById("sector-filter");
  const regionSelect = document.getElementById("region-filter");
  const searchButton = document.getElementById("search-button");
  const searchButtonLabel = searchButton.querySelector(".button-label");
  const progress = document.getElementById("search-progress");
  const feedback = document.getElementById("search-feedback");
  const resultsRegion = document.getElementById("results");
  const sectorOptions = document.getElementById("sector-options");
  const regionOptions = document.getElementById("region-options");

  const addDialog = document.getElementById("add-dialog");
  const addForm = document.getElementById("add-form");
  const addError = document.getElementById("add-error");
  const addSubmit = document.getElementById("add-submit");
  const addSubmitLabel = addSubmit.querySelector(".button-label");
  const addedNote = document.getElementById("added-note");
  const addedNoteName = document.getElementById("added-note-name");
  const addedNoteText = document.getElementById("added-note-text");
  const directoryDialog = document.getElementById("directory-dialog");
  const directoryList = document.getElementById("directory-list");
  const directoryCount = document.getElementById("directory-count");
  const directorySubtitle = document.getElementById("directory-subtitle");
  const resetDemo = document.getElementById("reset-demo");
  const resetDemoLabel = resetDemo.querySelector(".button-label");
  const personDialog = document.getElementById("person-dialog");
  const personName = document.getElementById("person-dialog-name");
  const personRole = document.getElementById("person-dialog-role");
  const personBody = document.getElementById("person-dialog-body");

  let activeController = null;
  let lastQuery = "";
  let profilesById = {};

  /* ---------- safe DOM helpers (textContent only, never markup) ---------- */

  function el(tagName, className, text) {
    const node = document.createElement(tagName);
    if (className) {
      node.className = className;
    }
    if (text !== undefined && text !== null) {
      node.textContent = String(text);
    }
    return node;
  }

  function bdi(parent, value, className) {
    const node = el("bdi", className, value || "");
    node.dir = "auto";
    parent.appendChild(node);
    return node;
  }

  function clean(value) {
    if (value === undefined || value === null) {
      return "";
    }
    return String(value).trim();
  }

  /* ---------- metadata: filter selects + form datalists ---------- */

  function fillOptions(select, values) {
    const keep = select.options[0];
    select.replaceChildren(keep);
    values.forEach(function (value) {
      const cleanValue = clean(value);
      if (!cleanValue) {
        return;
      }
      const option = document.createElement("option");
      option.value = cleanValue;
      option.textContent = cleanValue;
      select.appendChild(option);
    });
  }

  function fillDatalist(datalist, values) {
    datalist.replaceChildren();
    values.forEach(function (value) {
      const cleanValue = clean(value);
      if (!cleanValue) {
        return;
      }
      const option = document.createElement("option");
      option.value = cleanValue;
      datalist.appendChild(option);
    });
  }

  async function loadMetadata() {
    try {
      const response = await fetch("/api/meta", { headers: { Accept: "application/json" } });
      if (!response.ok) {
        return;
      }
      const metadata = await response.json();
      const sectors = Array.isArray(metadata?.filters?.sectors) ? metadata.filters.sectors : [];
      const regions = Array.isArray(metadata?.filters?.regions) ? metadata.filters.regions : [];
      const currentSector = sectorSelect.value;
      const currentRegion = regionSelect.value;
      fillOptions(sectorSelect, sectors);
      fillOptions(regionSelect, regions);
      if (sectors.includes(currentSector)) {
        sectorSelect.value = currentSector;
      }
      if (regions.includes(currentRegion)) {
        regionSelect.value = currentRegion;
      }
      fillDatalist(sectorOptions, sectors);
      fillDatalist(regionOptions, regions);
    } catch (_error) {
      /* Metadata improves the form; free-text search works without it. */
    }
  }

  /* ---------- profile directory ---------- */

  const DETAIL_FIELDS = [
    ["experience", "ניסיון"],
    ["areas_of_activity", "תחומי פעילות"],
    ["interests", "תחומי עניין"],
    ["values", "ערכים"],
    ["affiliations", "השתייכויות ורשתות"],
    ["description", "תיאור חופשי"],
    ["cohort", "מחזור"]
  ];

  async function loadDirectory() {
    try {
      const response = await fetch("/api/profiles", { headers: { Accept: "application/json" } });
      if (!response.ok) {
        return null;
      }
      const payload = await response.json();
      const profiles = Array.isArray(payload.profiles) ? payload.profiles : [];
      profilesById = {};
      profiles.forEach(function (profile) {
        if (profile && profile.profile_id) {
          profilesById[profile.profile_id] = profile;
        }
      });
      directoryCount.textContent = String(profiles.length);
      directoryCount.hidden = !profiles.length;
      // Nothing to reset until someone has actually added a person, so the
      // control stays out of the way on a fresh index.
      resetDemo.hidden = !payload.added_count;
      return payload;
    } catch (_error) {
      return null;
    }
  }

  function appendDetailRows(parent, profile) {
    const list = el("dl", "person-details");
    DETAIL_FIELDS.forEach(function (pair) {
      const value = clean(profile[pair[0]]);
      if (!value) {
        return;
      }
      const term = el("dt", "", pair[1]);
      const detail = el("dd");
      bdi(detail, value);
      list.appendChild(term);
      list.appendChild(detail);
    });
    if (list.childNodes.length) {
      parent.appendChild(list);
    } else {
      parent.appendChild(el("p", "person-details-empty", "אין שדות תיאור נוספים בפרופיל זה."));
    }
  }

  function directoryRow(profile) {
    const row = el("details", "directory-row");
    const summary = el("summary");
    const head = el("div", "directory-row-head");
    const identity = el("div");
    const name = el("strong");
    const trigger = el("button", "name-trigger");
    trigger.type = "button";
    trigger.dataset.person = clean(profile.profile_id);
    trigger.title = "הצגת כרטיס איש קשר";
    bdi(trigger, clean(profile.name) || "פרופיל ללא שם");
    name.appendChild(trigger);
    identity.appendChild(name);
    if (profile.added) {
      identity.appendChild(el("span", "added-badge", "נוסף בהדגמה"));
    }
    const role = clean(profile.title);
    const organisation = clean(profile.organisation);
    if (role || organisation) {
      const line = el("p", "role-line");
      if (role) {
        bdi(line, role);
      }
      if (role && organisation) {
        line.appendChild(document.createTextNode(" · "));
      }
      if (organisation) {
        bdi(line, organisation);
      }
      identity.appendChild(line);
    }
    head.appendChild(identity);
    const tags = el("div", "card-tags");
    [profile.sector, profile.region].forEach(function (value) {
      const cleanValue = clean(value);
      if (cleanValue) {
        const tag = el("span", "tag");
        bdi(tag, cleanValue);
        tags.appendChild(tag);
      }
    });
    if (tags.childNodes.length) {
      head.appendChild(tags);
    }
    summary.appendChild(head);
    row.appendChild(summary);
    const body = el("div", "directory-row-body");
    appendDetailRows(body, profile);
    row.appendChild(body);
    return row;
  }

  /* ---------- person contact card ---------- */

  function openPersonCard(profileId) {
    const profile = profilesById[clean(profileId)];
    if (!profile) {
      return;
    }
    personName.textContent = "";
    bdi(personName, clean(profile.name) || "פרופיל ללא שם");
    if (profile.added) {
      personName.appendChild(el("span", "added-badge", "נוסף בהדגמה"));
    }

    personRole.textContent = "";
    const role = clean(profile.title);
    const organisation = clean(profile.organisation);
    if (role) {
      bdi(personRole, role);
    }
    if (role && organisation) {
      personRole.appendChild(document.createTextNode(" · "));
    }
    if (organisation) {
      bdi(personRole, organisation);
    }

    const nodes = [];
    const contact = el("div", "contact-block");
    contact.appendChild(el("h3", "", "פרטי קשר"));
    const contactList = el("dl", "person-details");
    [["email", "אימייל"], ["phone", "טלפון"]].forEach(function (pair) {
      const value = clean(profile[pair[0]]);
      contactList.appendChild(el("dt", "", pair[1]));
      const detail = el("dd");
      if (value) {
        // Rendered as text, not a mailto:/tel: link: these addresses are
        // fictional, and a link that opens a mail composer to nowhere is worse
        // than plain text.
        bdi(detail, value);
      } else {
        detail.appendChild(el("span", "contact-missing", "לא הוזן"));
      }
      contactList.appendChild(detail);
    });
    contact.appendChild(contactList);
    nodes.push(contact);

    const tags = el("div", "card-tags");
    [profile.sector, profile.region].forEach(function (value) {
      const cleanValue = clean(value);
      if (cleanValue) {
        const tag = el("span", "tag");
        bdi(tag, cleanValue);
        tags.appendChild(tag);
      }
    });
    if (tags.childNodes.length) {
      nodes.push(tags);
    }

    const details = el("div", "person-card-details");
    details.appendChild(el("h3", "", "מתוך הפרופיל"));
    appendDetailRows(details, profile);
    nodes.push(details);

    personBody.replaceChildren.apply(personBody, nodes);
    if (directoryDialog.open) {
      directoryDialog.close();
    }
    personDialog.showModal();
  }

  function renderDirectory(payload) {
    if (!payload) {
      directoryList.replaceChildren(el("p", "directory-loading", "רשימת הפרופילים אינה זמינה כרגע."));
      return;
    }
    const profiles = Array.isArray(payload.profiles) ? payload.profiles : [];
    // Hebrew does not take a plural noun or verb after 1, so the added-count
    // clause is built per number rather than interpolated into one string.
    const addedClause = payload.added_count === 1
      ? "מהם אחד שנוסף בהדגמה"
      : "מהם " + payload.added_count + " שנוספו בהדגמה";
    directorySubtitle.textContent = payload.added_count
      ? profiles.length + " פרופילים באינדקס, " + addedClause + ". תוספות נשמרות בזיכרון עד לכיבוי השרת."
      : profiles.length + " פרופילים באינדקס. אפשר להוסיף משלכם ולראות אותם כאן.";
    const nodes = profiles.map(directoryRow);
    // Added profiles float to the top: they are what the visitor came to check.
    nodes.sort(function (a, b) {
      const addedA = a.querySelector(".added-badge") ? 1 : 0;
      const addedB = b.querySelector(".added-badge") ? 1 : 0;
      return addedB - addedA;
    });
    directoryList.replaceChildren.apply(directoryList, nodes);
  }

  async function openDirectory() {
    if (addDialog.open) {
      addDialog.close();
    }
    directoryList.replaceChildren(el("p", "directory-loading", "טוען את רשימת הפרופילים…"));
    directoryDialog.showModal();
    renderDirectory(await loadDirectory());
  }

  function closeDirectory() {
    directoryDialog.close();
  }

  // Returns the index to the sealed corpus.  The dialog is already open when this
  // runs, so it re-renders in place rather than calling showModal() a second time.
  async function runDemoReset() {
    resetDemo.disabled = true;
    resetDemoLabel.textContent = "מאפס…";
    try {
      const response = await fetch("/api/reset", {
        method: "POST",
        headers: { Accept: "application/json" }
      });
      if (!response.ok) {
        throw new Error("איפוס ההדגמה נכשל");
      }
      const body = await response.json();
      addedNote.hidden = true;
      setFeedback(clean(body.message));
      loadMetadata();
      renderDirectory(await loadDirectory());
    } catch (_error) {
      setFeedback("איפוס ההדגמה נכשל; אפשר לנסות שוב");
    } finally {
      resetDemoLabel.textContent = "איפוס ההדגמה";
      resetDemo.disabled = false;
    }
  }

  /* ---------- search rendering ---------- */

  function setFeedback(message) {
    feedback.textContent = message;
  }

  function setLoading(isLoading) {
    resultsRegion.setAttribute("aria-busy", String(isLoading));
    searchButton.disabled = isLoading;
    searchButtonLabel.textContent = isLoading ? "מחפש…" : "חיפוש";
    progress.hidden = !isLoading;
    document.querySelectorAll(".example-chip").forEach(function (chip) {
      chip.disabled = isLoading;
    });
  }

  function resultsHeading(query, count) {
    const wrapper = el("div", "results-heading");
    const title = el("h2");
    title.id = "results-heading";
    title.tabIndex = -1;
    title.appendChild(document.createTextNode("התאמות עבור "));
    bdi(title, "„" + query + "”");
    wrapper.appendChild(title);
    wrapper.appendChild(el("p", "", count === 1 ? "התאמה אחת" : count + " התאמות"));
    return { wrapper: wrapper, title: title };
  }

  function tierBadge(tier) {
    const strong = clean(tier) === "חזקה";
    const badge = el("span", "tier", strong ? "התאמה חזקה" : "התאמה אפשרית");
    badge.dataset.tier = strong ? "strong" : "possible";
    return badge;
  }

  function highlightInto(parent, text, needle) {
    const source = clean(text);
    const phrase = clean(needle);
    if (!source) {
      return;
    }
    if (!phrase) {
      parent.appendChild(document.createTextNode(source));
      return;
    }
    const at = source.toLocaleLowerCase("he").indexOf(phrase.toLocaleLowerCase("he"));
    if (at < 0) {
      parent.appendChild(document.createTextNode(source));
      return;
    }
    parent.appendChild(document.createTextNode(source.slice(0, at)));
    parent.appendChild(el("mark", "", source.slice(at, at + phrase.length)));
    parent.appendChild(document.createTextNode(source.slice(at + phrase.length)));
  }

  function resultCard(result) {
    const safe = result && typeof result === "object" ? result : {};
    const card = el("article", "result-card");

    const top = el("div", "card-top");
    const identity = el("div");
    const name = el("h3");
    const profileId = clean(safe.profile_id);
    if (profilesById[profileId]) {
      // A real button, not a styled span: keyboard and screen-reader users get
      // the same affordance as a mouse click.
      const trigger = el("button", "name-trigger");
      trigger.type = "button";
      trigger.dataset.person = profileId;
      trigger.title = "הצגת כרטיס איש קשר";
      bdi(trigger, clean(safe.name) || "פרופיל ללא שם");
      name.appendChild(trigger);
    } else {
      bdi(name, clean(safe.name) || "פרופיל ללא שם");
    }
    identity.appendChild(name);
    const role = clean(safe.title);
    const organisation = clean(safe.organisation);
    if (role || organisation) {
      const line = el("p", "role-line");
      if (role) {
        bdi(line, role);
      }
      if (role && organisation) {
        line.appendChild(document.createTextNode(" · "));
      }
      if (organisation) {
        bdi(line, organisation);
      }
      identity.appendChild(line);
    }
    top.appendChild(identity);
    top.appendChild(tierBadge(safe.confidence_tier));
    card.appendChild(top);

    const tags = el("div", "card-tags");
    [safe.sector, safe.region].forEach(function (value) {
      const cleanValue = clean(value);
      if (cleanValue) {
        const tag = el("span", "tag");
        bdi(tag, cleanValue);
        tags.appendChild(tag);
      }
    });
    if (safe.semantic_only) {
      tags.appendChild(el("span", "tag tag--accent", "ללא מילים משותפות עם החיפוש"));
    }
    if (tags.childNodes.length) {
      card.appendChild(tags);
    }

    const evidenceText = clean(safe.evidence_span);
    if (evidenceText) {
      const evidence = el("div", "evidence");
      const quote = el("blockquote");
      highlightInto(quote, evidenceText, safe.evidence_highlight);
      evidence.appendChild(quote);
      const meta = el("div", "evidence-meta");
      const aspect = clean(safe.winning_aspect_label);
      if (aspect) {
        meta.appendChild(el("span", "", "מתוך: " + aspect));
      }
      const provenance = clean(safe.provenance);
      if (provenance) {
        meta.appendChild(el("span", "", provenance));
      }
      if (meta.childNodes.length) {
        evidence.appendChild(meta);
      }
      card.appendChild(evidence);
    }

    const bridge = clean(safe.concept_bridge);
    if (bridge) {
      const line = el("p", "bridge-line");
      line.appendChild(el("strong", "", "גשר מושגי"));
      line.appendChild(document.createTextNode(" "));
      bdi(line, bridge);
      card.appendChild(line);
    }

    // Full person details, joined client-side from the directory payload so the
    // search response itself stays a ranking contract, not a record dump.
    const fullProfile = profilesById[clean(safe.profile_id)];
    if (fullProfile) {
      const more = el("details", "card-more");
      more.appendChild(el("summary", "", "הפרופיל המלא"));
      const body = el("div", "card-more-body");
      appendDetailRows(body, fullProfile);
      more.appendChild(body);
      card.appendChild(more);
    }

    return card;
  }

  function renderResults(payload) {
    const results = Array.isArray(payload.results) ? payload.results.slice(0, 5) : [];
    const query = clean(payload.query) || lastQuery;
    if (!results.length) {
      renderNoMatch(query);
      return;
    }
    const heading = resultsHeading(query, results.length);
    const nodes = [heading.wrapper];
    results.forEach(function (result) {
      nodes.push(resultCard(result));
    });
    resultsRegion.replaceChildren.apply(resultsRegion, nodes);
    setFeedback(clean(payload.message) || "החיפוש הושלם");
    heading.title.focus({ preventScroll: true });
  }

  function renderNoMatch(query) {
    const wrapper = el("div", "empty-state");
    wrapper.appendChild(el("h2", "", "לא נמצאה התאמה חזקה"));
    const copy = el("p");
    copy.appendChild(document.createTextNode("עבור "));
    bdi(copy, "„" + query + "”");
    copy.appendChild(document.createTextNode(" אף פרופיל לא עבר את סף האמון, והמערכת מעדיפה לומר זאת במקום להציג תוצאות חלשות."));
    wrapper.appendChild(copy);
    // Measured behaviour, not a guess: very short queries often score inside the
    // band where out-of-domain noise also lives, while a fuller clause on the same
    // need clears the gate (e.g. עזרה לבעלי חיים at 0.60 vs פעיל למען חיות at 0.47).
    // The hint appears only for short queries so the honest-refusal demo on a long
    // out-of-domain query keeps its clean message.
    if (query.split(/\s+/).filter(Boolean).length <= 3) {
      wrapper.appendChild(el(
        "p",
        "empty-hint",
        "שאילתות של מילה־שתיים נעצרות לעיתים בסף גם כשיש התאמה. נסחו את הצורך במשפט מלא יותר — תחום, תפקיד או הקשר — והדיוק משתפר."
      ));
    }
    resultsRegion.replaceChildren(wrapper);
    setFeedback("לא נמצאה התאמה חזקה");
  }

  function renderError(message) {
    const wrapper = el("div", "error-state");
    wrapper.setAttribute("role", "alert");
    wrapper.appendChild(el("h2", "", "החיפוש לא הושלם"));
    wrapper.appendChild(el("p", "", clean(message) || "אירעה תקלה מקומית; אפשר לנסות שוב."));
    resultsRegion.replaceChildren(wrapper);
    setFeedback("החיפוש נכשל");
  }

  async function performSearch(overrideQuery) {
    const query = clean(overrideQuery !== undefined ? overrideQuery : queryInput.value);
    if (!query) {
      queryInput.focus();
      setFeedback("יש לכתוב שאלה או תיאור לפני החיפוש");
      return;
    }
    if (overrideQuery !== undefined) {
      queryInput.value = query;
    }
    lastQuery = query;

    if (activeController) {
      activeController.abort();
    }
    const controller = new AbortController();
    activeController = controller;
    setLoading(true);
    setFeedback("מחפש…");

    try {
      const response = await fetch("/api/search", {
        method: "POST",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify({
          query: query,
          filters: { sector: sectorSelect.value, region: regionSelect.value }
        }),
        signal: controller.signal
      });
      let payload = {};
      try {
        payload = await response.json();
      } catch (_error) {
        payload = {};
      }
      if (!response.ok) {
        throw new Error(clean(payload.message) || "שירות החיפוש המקומי אינו זמין כרגע");
      }
      if (clean(payload.status) === "no_strong_match") {
        renderNoMatch(clean(payload.query) || query);
      } else {
        renderResults(payload);
      }
    } catch (error) {
      if (error.name !== "AbortError") {
        renderError(error.message);
      }
    } finally {
      if (activeController === controller) {
        activeController = null;
        setLoading(false);
      }
    }
  }

  /* ---------- add-profile dialog ---------- */

  function openAddDialog() {
    if (directoryDialog.open) {
      directoryDialog.close();
    }
    addError.hidden = true;
    addDialog.showModal();
    document.getElementById("add-first-name").focus();
  }

  function closeAddDialog() {
    addDialog.close();
  }

  function setAddLoading(isLoading) {
    addSubmit.disabled = isLoading;
    addSubmitLabel.textContent = isLoading ? "מקודד את הפרופיל…" : "הוספה לאינדקס";
  }

  function addFormPayload() {
    const value = function (id) {
      return clean(document.getElementById(id).value);
    };
    return {
      first_name: value("add-first-name"),
      last_name: value("add-last-name"),
      title: value("add-title"),
      organisation: value("add-organisation"),
      sector: value("add-sector"),
      region: value("add-region"),
      cohort: value("add-cohort"),
      experience: value("add-experience"),
      areas_of_activity: value("add-areas"),
      interests: value("add-interests"),
      values: value("add-values"),
      affiliations: value("add-affiliations"),
      description: value("add-description"),
      email: value("add-email"),
      phone: value("add-phone")
    };
  }

  async function submitAddForm(event) {
    event.preventDefault();
    const payload = addFormPayload();
    addError.hidden = true;

    if (!payload.first_name || !payload.last_name) {
      addError.textContent = "נדרשים שם פרטי ושם משפחה";
      addError.hidden = false;
      return;
    }
    const hasContent = [
      payload.title, payload.organisation, payload.experience,
      payload.areas_of_activity, payload.interests, payload.values,
      payload.affiliations, payload.description
    ].some(Boolean);
    if (!hasContent) {
      addError.textContent = "נדרש תוכן באחד משדות התיאור לפחות — אחרת אין מה לחפש";
      addError.hidden = false;
      return;
    }

    setAddLoading(true);
    try {
      const response = await fetch("/api/profiles", {
        method: "POST",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      let body = {};
      try {
        body = await response.json();
      } catch (_error) {
        body = {};
      }
      if (!response.ok) {
        throw new Error(clean(body.message) || "ההוספה נכשלה; אפשר לנסות שוב");
      }
      addForm.reset();
      closeAddDialog();
      addedNoteName.textContent = clean(body.name);
      addedNoteText.textContent = "נוסף לאינדקס. החיפוש מאתר לפי תוכן — נסחו את הצורך במילים שלכם ובדקו אם הפרופיל עולה.";
      addedNote.hidden = false;
      loadMetadata();
      loadDirectory();
      queryInput.focus();
    } catch (error) {
      addError.textContent = clean(error.message);
      addError.hidden = false;
    } finally {
      setAddLoading(false);
    }
  }

  /* ---------- wiring ---------- */

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    performSearch();
  });

  document.addEventListener("click", function (event) {
    const example = event.target.closest("[data-example]");
    if (example && !example.disabled) {
      performSearch(example.dataset.example);
      return;
    }
    const person = event.target.closest("[data-person]");
    if (person) {
      openPersonCard(person.dataset.person);
    }
  });

  document.getElementById("open-add-dialog").addEventListener("click", openAddDialog);
  document.getElementById("open-add-dialog-inline").addEventListener("click", openAddDialog);
  document.getElementById("close-add-dialog").addEventListener("click", closeAddDialog);
  document.getElementById("cancel-add-dialog").addEventListener("click", closeAddDialog);
  document.getElementById("open-directory").addEventListener("click", openDirectory);
  document.getElementById("close-directory").addEventListener("click", closeDirectory);
  resetDemo.addEventListener("click", runDemoReset);
  document.getElementById("close-person").addEventListener("click", function () {
    personDialog.close();
  });
  addForm.addEventListener("submit", submitAddForm);
  document.getElementById("added-note-search").addEventListener("click", function () {
    queryInput.focus();
    queryInput.select();
  });

  loadMetadata();
  loadDirectory();
})();
