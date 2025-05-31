---
layout: page
title: Experience
permalink: /experience
weight: 2
---

<style>
.toggle-buttons {
  margin-bottom: 1rem;
  display: flex;
  justify-content: center;
}

.toggle-buttons button {
  margin-right: 10px;
  padding: 8px 16px;
  font-size: 1rem;
  cursor: pointer;
}

.experience-section {
  display: none;
}

.experience-section.active {
  display: block;
}
</style>

<div class="toggle-buttons">
  <button type="button" class="btn btn-outline-primary" onclick="showSection('work')">Work Experience</button>
  <button type="button" class="btn btn-outline-primary" onclick="showSection('volunteer')">Volunteer Experience</button>
</div>

<div id="work" class="experience-section active">
  <h3><strong>Work Experience</strong></h3>
  <div class="row">
    {% include about/work_timeline.html %}
  </div>
</div>

<div id="volunteer" class="experience-section">
  <h3><strong>Volunteer Experience</strong></h3>
  <div class="row">
    {% include about/volunteer_timeline.html %}
  </div>
</div>

<script>
function showSection(sectionId) {
  const sections = document.querySelectorAll('.experience-section');
  sections.forEach(section => section.classList.remove('active'));
  document.getElementById(sectionId).classList.add('active');
}
</script>