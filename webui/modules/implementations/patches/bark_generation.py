import bark.generation as o
from bark.generation import *


SUPPORTED_LANGS = [
    ("English", "en"),
    ("German", "de"),
    ("Spanish", "es"),
    ("French", "fr"),
    ("Hindi", "hi"),
    ("Italian", "it"),
    ("Japanese", "ja"),
    ("Korean", "ko"),
    ("Polish", "pl"),
    ("Portuguese", "pt"),
    ("Russian", "ru"),
    ("Turkish", "tr"),
    ("Chinese", "zh"),
]

ALLOWED_PROMPTS = {"announcer"}
for _, lang in SUPPORTED_LANGS:
    for n in range(10):
        ALLOWED_PROMPTS.add(f"{lang}_speaker_{n}")
for n in range(10):
    ALLOWED_PROMPTS.add(f"speaker_{n}")

# TODO: generate_course and generate_fine


def generate_text_semantic_new(
    text,
    history_prompt=None,
    temp=0.7,
    top_k=None,
    top_p=None,
    silent=False,
    min_eos_p=0.2,
    max_gen_duration_s=None,
    allow_early_stop=True,
    use_kv_caching=False,
):
    """Generate semantic tokens from text."""
    assert isinstance(text, str)
    text = o._normalize_whitespace(text)
    assert len(text.strip()) > 0
    if history_prompt is not None:
        skip = False
        if history_prompt.endswith(".npz"):
            semantic_history = np.load(history_prompt)["semantic_prompt"]
        else:
            if history_prompt in ALLOWED_PROMPTS:
                semantic_history = np.load(
                    os.path.join(CUR_PATH, "assets", "prompts", f"{history_prompt}.npz")
                )["semantic_prompt"]
            else:
                filename = f'data/bark_custom_speakers/{history_prompt}.npz'
                if os.path.isfile(filename):
                    semantic_history = np.load(
                        filename
                    )["semantic_prompt"]
                else:
                    skip = True
        if not skip:
            assert (
                isinstance(semantic_history, np.ndarray)
                and len(semantic_history.shape) == 1
                and len(semantic_history) > 0
                and semantic_history.min() >= 0
                and semantic_history.max() <= SEMANTIC_VOCAB_SIZE - 1
            )
        else:
            semantic_history = None
    else:
        semantic_history = None
    # load models if not yet exist
    global models
    global models_devices
    if "text" not in models:
        preload_models()
    model_container = models["text"]
    model = model_container["model"]
    tokenizer = model_container["tokenizer"]
    encoded_text = np.array(o._tokenize(tokenizer, text)) + TEXT_ENCODING_OFFSET
    if OFFLOAD_CPU:
        model.to(models_devices["text"])
    device = next(model.parameters()).device
    if len(encoded_text) > 256:
        p = round((len(encoded_text) - 256) / len(encoded_text) * 100, 1)
        logger.warning(f"warning, text too long, lopping of last {p}%")
        encoded_text = encoded_text[:256]
    encoded_text = np.pad(
        encoded_text,
        (0, 256 - len(encoded_text)),
        constant_values=TEXT_PAD_TOKEN,
        mode="constant",
    )
    if semantic_history is not None:
        semantic_history = semantic_history.astype(np.int64)
        # lop off if history is too long, pad if needed
        semantic_history = semantic_history[-256:]
        semantic_history = np.pad(
            semantic_history,
            (0, 256 - len(semantic_history)),
            constant_values=SEMANTIC_PAD_TOKEN,
            mode="constant",
        )
    else:
        semantic_history = np.array([SEMANTIC_PAD_TOKEN] * 256)
    x = torch.from_numpy(
        np.hstack([
            encoded_text, semantic_history, np.array([SEMANTIC_INFER_TOKEN])
        ]).astype(np.int64)
    )[None]
    assert x.shape[1] == 256 + 256 + 1
    with o._inference_mode():
        x = x.to(device)
        n_tot_steps = 768
        # custom tqdm updates since we don't know when eos will occur
        pbar = tqdm.tqdm(disable=silent, total=100)
        pbar_state = 0
        tot_generated_duration_s = 0
        kv_cache = None
        for n in range(n_tot_steps):
            if use_kv_caching and kv_cache is not None:
                x_input = x[:, [-1]]
            else:
                x_input = x
            logits, kv_cache = model(
                x_input, merge_context=True, use_cache=use_kv_caching, past_kv=kv_cache
            )
            relevant_logits = logits[0, 0, :SEMANTIC_VOCAB_SIZE]
            if allow_early_stop:
                relevant_logits = torch.hstack(
                    (relevant_logits, logits[0, 0, [SEMANTIC_PAD_TOKEN]])  # eos
                )
            if top_p is not None:
                # faster to convert to numpy
                logits_device = relevant_logits.device
                logits_dtype = relevant_logits.type()
                relevant_logits = relevant_logits.detach().cpu().type(torch.float32).numpy()
                sorted_indices = np.argsort(relevant_logits)[::-1]
                sorted_logits = relevant_logits[sorted_indices]
                cumulative_probs = np.cumsum(softmax(sorted_logits))
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].copy()
                sorted_indices_to_remove[0] = False
                relevant_logits[sorted_indices[sorted_indices_to_remove]] = -np.inf
                relevant_logits = torch.from_numpy(relevant_logits)
                relevant_logits = relevant_logits.to(logits_device).type(logits_dtype)
            if top_k is not None:
                v, _ = torch.topk(relevant_logits, min(top_k, relevant_logits.size(-1)))
                relevant_logits[relevant_logits < v[-1]] = -float("Inf")
            probs = F.softmax(relevant_logits / temp, dim=-1)
            # multinomial bugged on mps: shuttle to cpu if necessary
            inf_device = probs.device
            if probs.device.type == "mps":
                probs = probs.to("cpu")
            item_next = torch.multinomial(probs, num_samples=1)
            probs = probs.to(inf_device)
            item_next = item_next.to(inf_device)
            if allow_early_stop and (
                item_next == SEMANTIC_VOCAB_SIZE
                or (min_eos_p is not None and probs[-1] >= min_eos_p)
            ):
                # eos found, so break
                pbar.update(100 - pbar_state)
                break
            x = torch.cat((x, item_next[None]), dim=1)
            tot_generated_duration_s += 1 / SEMANTIC_RATE_HZ
            if max_gen_duration_s is not None and tot_generated_duration_s > max_gen_duration_s:
                pbar.update(100 - pbar_state)
                break
            if n == n_tot_steps - 1:
                pbar.update(100 - pbar_state)
                break
            del logits, relevant_logits, probs, item_next
            req_pbar_state = np.min([100, int(round(100 * n / n_tot_steps))])
            if req_pbar_state > pbar_state:
                pbar.update(req_pbar_state - pbar_state)
            pbar_state = req_pbar_state
        pbar.close()
        out = x.detach().cpu().numpy().squeeze()[256 + 256 + 1 :]
    if OFFLOAD_CPU:
        model.to("cpu")
    assert all(0 <= out) and all(out < SEMANTIC_VOCAB_SIZE)
    o._clear_cuda_cache()
    return out
