const express = require('express');
const fetch = require('node-fetch');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3030;
const OLLAMA_URL = process.env.OLLAMA_URL || 'http://host.docker.internal:11434';

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// ── Health check ──────────────────────────────────────────────────────────────
app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', ollama: OLLAMA_URL });
});

// ── List Ollama models ────────────────────────────────────────────────────────
app.get('/api/models', async (req, res) => {
  try {
    const response = await fetch(`${OLLAMA_URL}/api/tags`);
    if (!response.ok) throw new Error(`Ollama responded with ${response.status}`);
    const data = await response.json();
    const models = (data.models || []).map(m => ({
      name: m.name,
      size: m.size,
      modified: m.modified_at
    }));
    res.json({ models });
  } catch (err) {
    console.error('Error fetching models:', err.message);
    res.status(502).json({ error: `Cannot reach Ollama at ${OLLAMA_URL}: ${err.message}` });
  }
});

// ── Field definitions for /api/analyze ───────────────────────────────────────
const FIELD_KEYS = {
  // Text frameworks
  costar:        ['context', 'objective', 'style', 'tone', 'audience', 'response_format'],
  risen:         ['role', 'instructions', 'steps', 'end_goal', 'narrowing'],
  rtf:           ['role', 'task', 'format'],
  cot:           ['role', 'problem', 'reasoning_hint', 'output_format'],
  fewshot:       ['role', 'task', 'example_input', 'example_output', 'format'],
  react:         ['role', 'problem', 'available_actions', 'output_format'],
  // Visual image
  visual_image:  ['subject', 'setting', 'art_style', 'lighting', 'camera', 'mood', 'color_palette', 'quality_tags', 'negative_prompt'],
  // Visual video
  visual_video:  ['scene', 'subject_motion', 'camera_movement', 'shot_type', 'lighting', 'mood_style', 'audio_cue', 'duration_pacing']
};

// ── Language instruction helper ───────────────────────────────────────────────
function addLangInstruction(systemPrompt, outputLang, isVisual = false) {
  if (outputLang !== 'es') return systemPrompt;
  const visualNote = isVisual
    ? ' Exception: keep all platform-specific technical syntax terms in English (e.g. --ar, --v, masterpiece, dolly, pan, zoom, etc.) as these are required keywords for the target AI generator.'
    : '';
  return systemPrompt + `\n\n- LANGUAGE REQUIREMENT: Write your entire response in Spanish (Español). Every word of the output must be in Spanish.${visualNote}`;
}

// ── Analyze raw input → structured field suggestions (non-streaming) ──────────
app.post('/api/analyze', async (req, res) => {
  const { rawPrompt, framework, model, outputLang } = req.body;
  if (!rawPrompt || !model) {
    return res.status(400).json({ error: 'rawPrompt and model are required' });
  }

  const fields = FIELD_KEYS[framework] || FIELD_KEYS.costar;
  const isVisual = framework.startsWith('visual_');
  const contextHint = isVisual
    ? `You are an expert AI art director and visual prompt engineer.`
    : `You are a prompt engineering expert.`;

  const baseSystemPrompt = `${contextHint} Given a rough user request, extract and infer the best values for each field listed below. Return ONLY a valid JSON object with these exact keys: ${fields.join(', ')}.

Rules:
- Infer missing context intelligently from the user's intent
- Keep each value concise but complete (1-3 sentences max per field, or a comma-separated tag list for visual fields)
- If a field cannot be reasonably inferred, use an empty string ""
- Do not include any explanation, preamble, or markdown fences — output raw JSON only`;

  const systemPrompt = addLangInstruction(baseSystemPrompt, outputLang, isVisual);

  const userMessage = `Extract fields from this rough request (framework: ${framework}):\n\n${rawPrompt}`;

  try {
    const ollamaRes = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        stream: false,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userMessage }
        ],
        options: { temperature: 0.3, top_p: 0.9 }
      })
    });

    if (!ollamaRes.ok) {
      const errText = await ollamaRes.text();
      return res.status(502).json({ error: errText });
    }

    const data = await ollamaRes.json();
    const raw = (data.message?.content || '').trim();
    const cleaned = raw.replace(/^```[a-z]*\n?/i, '').replace(/```$/i, '').trim();

    try {
      res.json({ fields: JSON.parse(cleaned) });
    } catch {
      res.json({ fields: {}, raw_fallback: cleaned });
    }

  } catch (err) {
    console.error('Analyze error:', err.message);
    res.status(502).json({ error: err.message });
  }
});

// ══════════════════════════════════════════════════════════════════════════════
// PROMPT ASSEMBLY — TEXT FRAMEWORKS
// ══════════════════════════════════════════════════════════════════════════════
function assembleFrameworkSystemPrompt(framework, style) {
  const styleGuides = {
    precise:    'Emphasize clarity, specificity, and technical completeness.',
    creative:   'Encourage exploration, imagination, and open-ended thinking.',
    concise:    'Optimize for brevity — strip all fluff, every word must earn its place.',
    stepbystep: 'Emphasize ordered process and numbered, explicit instructions.',
    debug:      'Emphasize diagnostic steps, error context, and environment details.'
  };

  const styleHint = styleGuides[style] || styleGuides.precise;

  const descriptions = {
    costar:  'COSTAR (Context · Objective · Style · Tone · Audience · Response Format) — full behavioral control for business, instructional, and content prompts.',
    risen:   'RISEN (Role · Instructions · Steps · End Goal · Narrowing) — agentic, task-driven prompts with a clear persona and scoped constraints.',
    rtf:     'RTF (Role · Task · Format) — lean and direct; best for narrow-scope, well-defined output prompts.',
    cot:     'Chain of Thought (Role · Problem · Reasoning Hint · Output Format) — forces step-by-step reasoning before answering.',
    fewshot: 'Few-Shot (Role · Task · Example Input/Output · Format) — demonstrates desired behavior with worked examples.',
    react:   'ReAct (Role · Problem · Available Actions · Output Format) — Reasoning + Acting with Thought/Action/Observation loops.'
  };

  return `You are a master prompt engineer. Assemble the provided structured components into a single, polished, immediately usable LLM prompt using the ${framework.toUpperCase()} framework.

Framework: ${descriptions[framework] || descriptions.costar}

Assembly rules:
- Output ONLY the assembled final prompt — no explanation, no meta-commentary
- The prompt must be self-contained; anyone can paste it directly into any LLM
- Weave components into natural prose — avoid raw "KEY: value" lists unless the framework demands labeled sections
- Include all non-empty fields; omit empty ones gracefully
- Style guidance: ${styleHint}`;
}

function buildUserMessageFromFields(framework, fields, rawPrompt) {
  const fieldLines = Object.entries(fields)
    .filter(([, v]) => v && String(v).trim())
    .map(([k, v]) => `${k.toUpperCase().replace(/_/g, ' ')}: ${v}`)
    .join('\n');

  return `Assemble a complete ${framework.toUpperCase()} prompt from these components:\n\n${fieldLines}\n\nOriginal raw request (reference only): ${rawPrompt}`;
}

// ══════════════════════════════════════════════════════════════════════════════
// PROMPT ASSEMBLY — VISUAL IMAGE
// ══════════════════════════════════════════════════════════════════════════════
const IMAGE_PLATFORM_PROMPTS = {
  midjourney: `You are a Midjourney v6 prompt expert. Assemble a Midjourney image prompt from the provided visual components.

Midjourney syntax rules:
- Write a vivid, comma-separated natural language description (subject, setting, style, lighting, camera, mood)
- Use strong artistic quality descriptors: highly detailed, intricate, masterpiece, sharp focus
- Reference specific art movements, aesthetics, or cinematic styles when relevant (e.g. "Bauhaus design", "35mm film grain", "cinematic Kodak Portra")
- Append Midjourney parameters at the very end after two dashes: --ar [ratio] --style raw --v 6.1
- If aspect ratio is not specified, default to --ar 16:9 for landscapes, --ar 1:1 for portraits
- Do NOT include a negative prompt (Midjourney uses --no flag instead, append it if provided)
- Output ONLY the final Midjourney prompt string — nothing else`,

  dalle3: `You are a DALL-E 3 prompt expert. Assemble a DALL-E 3 image prompt from the provided visual components.

DALL-E 3 rules:
- Write in complete, descriptive natural language sentences — DALL-E 3 excels at detailed prose
- Be extremely specific about the subject, its appearance, and the setting
- Explicitly state the style: "a digital painting", "a photograph", "an oil painting", "a 3D render"
- Describe lighting, mood, and composition in prose
- Include style safety: use "in the style of impressionism" not specific living artists
- No parameter syntax — DALL-E 3 does not use flags
- Output ONLY the final DALL-E 3 prompt string — nothing else`,

  stablediffusion: `You are a Stable Diffusion / Flux prompt expert. Assemble an optimized prompt from the provided visual components.

SD/Flux syntax rules:
- Format as comma-separated descriptive tags (not full sentences)
- Lead with the most important subject tags
- Use parentheses for emphasis: (masterpiece:1.3), (highly detailed:1.2), (sharp focus:1.1)
- Use square brackets to reduce weight: [blurry:0.5]
- Include quality boosters near the start: masterpiece, best quality, ultra detailed, 8k uhd
- After the positive prompt, on a new line write: Negative prompt: [comma-separated negative tags from the negative_prompt field, plus defaults: blurry, deformed, ugly, bad anatomy, extra limbs, watermark, text, low quality]
- Output ONLY the positive prompt, then a newline, then "Negative prompt: ..." — nothing else`,

  flux: `You are a Flux image model prompt expert (Black Forest Labs Flux.1). Assemble an optimized Flux prompt.

Flux-specific rules:
- Flux understands natural language better than classic SD — write clear descriptive sentences
- Include visual quality terms: "highly detailed", "professional photography", "8K resolution"
- Specify lighting explicitly: "dramatic side lighting", "soft diffused window light"
- State the medium: "photograph", "digital illustration", "oil painting", "3D render"
- Flux does not require heavy tag-weighting syntax — use it sparingly only for critical elements
- Include a negative prompt on a new line if negative_prompt is provided
- Output ONLY the final Flux prompt — nothing else`,

  firefly: `You are an Adobe Firefly prompt expert. Assemble a Firefly-optimized image prompt.

Firefly rules:
- Write in clear, descriptive natural language — Firefly excels at commercial and editorial imagery
- Firefly is copyright-safe: do NOT reference specific living artists, brands, or IP — use style descriptors instead (e.g. "in an impressionistic watercolor style" not "in the style of [artist]")
- Describe the style using: art movements, historical periods, photography styles, materials
- Include practical descriptors: color grading, lighting setups used in professional photography
- Specify intended use context if relevant: "commercial product photography", "editorial illustration"
- Output ONLY the final Firefly prompt string — nothing else`,

  universal: `You are a universal AI image prompt expert. Assemble an image generation prompt that works well across all major AI image generators (Midjourney, DALL-E 3, Stable Diffusion, Flux, Firefly, Ideogram).

Universal best practices:
- Write a rich, detailed description using natural language
- Structure: [subject and action] + [environment/setting] + [art style and medium] + [lighting] + [camera/composition] + [mood and atmosphere] + [quality descriptors]
- Include the visual style explicitly: "hyperrealistic photograph", "concept art illustration", "oil painting"
- Use universally understood photography and art terms
- Include quality hints that work everywhere: "highly detailed", "sharp focus", "professional quality"
- Output ONLY the final image prompt string — nothing else`
};

function buildImageSystemPrompt(platform) {
  return IMAGE_PLATFORM_PROMPTS[platform] || IMAGE_PLATFORM_PROMPTS.universal;
}

function buildImageUserMessage(platform, fields, rawIntent) {
  const fieldLines = Object.entries(fields)
    .filter(([, v]) => v && String(v).trim())
    .map(([k, v]) => `${k.toUpperCase().replace(/_/g, ' ')}: ${v}`)
    .join('\n');

  return `Assemble a ${platform.toUpperCase()} image generation prompt from these visual components:\n\n${fieldLines}\n\nOriginal intent (reference only): ${rawIntent}`;
}

// ══════════════════════════════════════════════════════════════════════════════
// PROMPT ASSEMBLY — VISUAL VIDEO
// ══════════════════════════════════════════════════════════════════════════════
const VIDEO_PLATFORM_PROMPTS = {
  sora: `You are an OpenAI Sora video prompt expert. Assemble a Sora video generation prompt from the provided components.

Sora prompt rules:
- Write a highly detailed, cinematic scene description in natural language
- Always specify camera behavior: "The camera slowly dollies forward", "A wide aerial shot pans left"
- Describe subject motion precisely: verbs, speed, direction (e.g. "a woman walks briskly through", "leaves flutter gently in the wind")
- Include atmospheric details: time of day, weather, light quality
- Specify the visual aesthetic: "cinematic 35mm film", "documentary handheld", "studio commercial"
- Sora handles longer, richer descriptions well — be thorough but purposeful
- Output ONLY the final Sora prompt paragraph — nothing else`,

  runway: `You are a Runway Gen-3 Alpha video prompt expert. Assemble a Runway video prompt from the provided components.

Runway Gen-3 rules:
- Keep the prompt concise and action-focused (aim for 100-300 words)
- Lead with the primary subject and their action: "A woman in a red coat walks through..."
- Camera motion must be explicit and precise: "camera slowly zooms in", "steady tracking shot", "handheld shaky cam"
- Include: subject, action, environment, camera motion, visual style
- Use cinematic quality terms: "cinematic", "4K", "shallow depth of field", "golden hour lighting"
- Runway is motion-sensitive — describe every moving element
- Output ONLY the final Runway prompt — nothing else`,

  pika: `You are a Pika video generation prompt expert. Assemble a Pika Labs video prompt from the provided components.

Pika rules:
- Write clear, action-driven descriptions — Pika responds well to motion verbs
- Structure: [subject] [action] [in/through/at setting] [camera motion] [visual style]
- Camera motion keywords that work well: "zoom in slowly", "pan right", "dolly out", "static shot"
- Include motion intensity: "gently", "rapidly", "smoothly", "dramatically"
- Style keywords: "cinematic", "photorealistic", "anime", "3D animated", "claymation"
- Keep descriptions punchy — Pika works well with focused, concrete descriptions
- Output ONLY the final Pika prompt string — nothing else`,

  kling: `You are a Kling AI video prompt expert. Assemble a Kling video generation prompt from the provided components.

Kling rules:
- Write in detailed, descriptive natural language — Kling handles long prompts well
- Describe the scene comprehensively: environment, time, atmosphere, lighting
- Specify subject movement with precision: direction, speed, and emotional quality
- Camera work: use standard film terminology (close-up, medium shot, wide shot, tracking, crane)
- Include tactile and sensory details that suggest motion: "fabric ripples", "water splashes", "breath visible in cold air"
- Kling excels at realistic human motion — describe body language and gesture
- Output ONLY the final Kling prompt — nothing else`,

  luma: `You are a Luma Dream Machine video prompt expert. Assemble a Luma video prompt from the provided components.

Luma Dream Machine rules:
- Luma excels at photorealistic, high-quality video — emphasize visual fidelity
- Lead with a strong scene-setting description: environment, time of day, light quality
- Describe lighting conditions in detail: "warm golden hour sun casts long shadows", "cool blue hour cityscape"
- Subject and motion: be specific but natural — Luma interprets motion fluidly
- Aesthetic keywords that work well: "photorealistic", "cinematic", "hyperdetailed", "film grain"
- Camera motion should feel intentional: "slow cinematic push in", "graceful aerial descent"
- Output ONLY the final Luma Dream Machine prompt — nothing else`,

  universal: `You are a universal AI video prompt expert. Assemble a video generation prompt that works across all major AI video generators (Sora, Runway, Pika, Kling, Luma, Hailuo).

Universal video prompt best practices:
- Structure: [scene/setting introduction] + [subject description] + [action and motion] + [camera movement and framing] + [lighting and atmosphere] + [visual style and quality]
- Always specify camera motion explicitly — this is the most important element for all platforms
- Use active, motion-focused language: present tense, strong verbs
- Include visual quality descriptors: "cinematic", "photorealistic", "sharp focus", "4K quality"
- Describe the pace and feel: "slow and dreamy", "fast-paced and energetic", "smooth and elegant"
- Keep it focused — one clear scene, one camera move, one emotional tone
- Output ONLY the final video prompt — nothing else`
};

function buildVideoSystemPrompt(platform) {
  return VIDEO_PLATFORM_PROMPTS[platform] || VIDEO_PLATFORM_PROMPTS.universal;
}

function buildVideoUserMessage(platform, fields, rawIntent) {
  const fieldLines = Object.entries(fields)
    .filter(([, v]) => v && String(v).trim())
    .map(([k, v]) => `${k.toUpperCase().replace(/_/g, ' ')}: ${v}`)
    .join('\n');

  return `Assemble a ${platform.toUpperCase()} video generation prompt from these components:\n\n${fieldLines}\n\nOriginal intent (reference only): ${rawIntent}`;
}

// ══════════════════════════════════════════════════════════════════════════════
// /api/reformat — SSE streaming endpoint (all modes)
// ══════════════════════════════════════════════════════════════════════════════
app.post('/api/reformat', async (req, res) => {
  const { rawPrompt, model, style, framework, fields, mode, platform, visualType, outputLang } = req.body;

  if (!rawPrompt || !model) {
    return res.status(400).json({ error: 'rawPrompt and model are required' });
  }

  let systemPrompt, userMessage;

  // ── Visual: Image ──────────────────────────────────────────────────────────
  if (mode === 'visual' && visualType === 'image') {
    const imgPlatform = platform || 'universal';
    systemPrompt = addLangInstruction(buildImageSystemPrompt(imgPlatform), outputLang, true);
    userMessage  = buildImageUserMessage(imgPlatform, fields || {}, rawPrompt);

  // ── Visual: Video ──────────────────────────────────────────────────────────
  } else if (mode === 'visual' && visualType === 'video') {
    const vidPlatform = platform || 'universal';
    systemPrompt = addLangInstruction(buildVideoSystemPrompt(vidPlatform), outputLang, true);
    userMessage  = buildVideoUserMessage(vidPlatform, fields || {}, rawPrompt);

  // ── Structured: text framework ─────────────────────────────────────────────
  } else if (mode === 'structured' && framework && fields && Object.keys(fields).length > 0) {
    systemPrompt = addLangInstruction(assembleFrameworkSystemPrompt(framework, style), outputLang);
    userMessage  = buildUserMessageFromFields(framework, fields, rawPrompt);

  // ── Quick: single-shot reformat ────────────────────────────────────────────
  } else {
    const styleGuides = {
      precise:    'Focus on clarity, specificity, and completeness.',
      creative:   'Encourage exploration and imaginative thinking.',
      concise:    'Optimize for brevity — strip all fluff.',
      stepbystep: 'Emphasize ordered process and numbered instructions.',
      debug:      'Emphasize diagnostic steps, error context, and environment details.'
    };

    systemPrompt = addLangInstruction(`You are an expert prompt engineer. Your sole job is to take a rough, incomplete, or ambiguous request and rewrite it as a single optimized prompt ready to paste into any LLM.

Rules:
- Output ONLY the improved prompt — no preamble, no explanation, no quotes
- Do not answer the question — only rewrite it as a better prompt
- Preserve the user's original intent exactly
- Add context, desired output format, tone, and constraints where helpful
- Eliminate vagueness and ambiguity
- Style guidance: ${styleGuides[style] || styleGuides.precise}`, outputLang);

    userMessage = `Rewrite this rough request as an optimized AI prompt:\n\n${rawPrompt}`;
  }

  // ── Stream via SSE ─────────────────────────────────────────────────────────
  try {
    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.flushHeaders();

    const ollamaRes = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        stream: true,
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userMessage }
        ],
        options: { temperature: 0.5, top_p: 0.9 }
      })
    });

    if (!ollamaRes.ok) {
      const errText = await ollamaRes.text();
      res.write(`data: ${JSON.stringify({ error: errText })}\n\n`);
      return res.end();
    }

    ollamaRes.body.on('data', chunk => {
      const lines = chunk.toString().split('\n').filter(l => l.trim());
      for (const line of lines) {
        try {
          const json = JSON.parse(line);
          if (json.message?.content) {
            res.write(`data: ${JSON.stringify({ token: json.message.content })}\n\n`);
          }
          if (json.done) {
            res.write(`data: ${JSON.stringify({ done: true })}\n\n`);
          }
        } catch (_) {}
      }
    });

    ollamaRes.body.on('end', () => res.end());
    ollamaRes.body.on('error', err => {
      res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
      res.end();
    });

  } catch (err) {
    console.error('Reformat error:', err.message);
    try {
      res.write(`data: ${JSON.stringify({ error: err.message })}\n\n`);
      res.end();
    } catch (_) {}
  }
});

app.listen(PORT, () => {
  console.log(`Prompt Forge v3 running on http://0.0.0.0:${PORT}`);
  console.log(`Ollama endpoint: ${OLLAMA_URL}`);
});
