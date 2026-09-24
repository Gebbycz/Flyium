import os
import re
import time
import subprocess
import re
import numpy as np
import pandas as pd


BASE = r"C:\Users\72553\Downloads\flyium\data"

CONNECTION_FILE = os.path.join(
    BASE, "connections_princeton.csv.gz"
)

NEURON_FILE = os.path.join(
    BASE, "neurons.csv.gz"
)

CELL_TYPE_FILE = os.path.join(
    BASE, "consolidated_cell_types.csv.gz"
)

CLASSIFICATION_FILE = os.path.join(
    BASE, "classification.csv.gz"
)

ACTIVE_NEURONS = 30000

PROPAGATION_STEPS = 8

ACTIVE_LIMIT = 12000

MIN_SYN_COUNT = 1

DECAY = 0.82

EXCITATORY_GAIN = 1.0
INHIBITORY_GAIN = 1.15


print()
print(" Starting connectome-driven fly...")
print()
print("Loading FAFB v783...")
print()


neurons = pd.read_csv(NEURON_FILE)

cell_types = pd.read_csv(CELL_TYPE_FILE)

classification = pd.read_csv(CLASSIFICATION_FILE)


connections = pd.read_csv(
    CONNECTION_FILE,
    usecols=[
        "pre_root_id",
        "post_root_id",
        "neuropil",
        "syn_count",
        "nt_type",
    ]
)

connections["pre_root_id"] = connections["pre_root_id"].astype(
    np.int64
)

connections["post_root_id"] = connections["post_root_id"].astype(
    np.int64
)

connections["syn_count"] = pd.to_numeric(
    connections["syn_count"],
    errors="coerce"
).fillna(1).astype(np.float32)

connections["nt_type"] = (
    connections["nt_type"]
    .fillna("UNKNOWN")
    .astype(str)
    .str.upper()
)

connections = connections[
    connections["syn_count"] >= MIN_SYN_COUNT
].copy()

print(
    f"Loaded {len(connections):,} FAFB synaptic records."
)


all_ids = np.unique(
    np.concatenate(
        [
            connections["pre_root_id"].to_numpy(),
            connections["post_root_id"].to_numpy(),
        ]
    )
)

print(
    f"Found {len(all_ids):,} connected neuron IDs."
)


print("Building sparse working network...")

degree_pre = (
    connections.groupby("pre_root_id")["syn_count"]
    .sum()
)

degree_post = (
    connections.groupby("post_root_id")["syn_count"]
    .sum()
)

degree = degree_pre.add(
    degree_post,
    fill_value=0
)

degree = degree.sort_values(
    ascending=False
)

seed_count = min(
    5000,
    len(degree)
)

seed_ids = list(
    degree.head(seed_count).index.astype(np.int64)
)

selected = set(seed_ids)


for _ in range(5):

    subset = connections[
        connections["pre_root_id"].isin(selected)
    ]

    candidates = (
        subset.groupby("post_root_id")["syn_count"]
        .sum()
        .sort_values(ascending=False)
    )

    for neuron_id in candidates.head(6000).index:

        selected.add(int(neuron_id))

        if len(selected) >= ACTIVE_NEURONS:
            break

    if len(selected) >= ACTIVE_NEURONS:
        break

working_ids = np.array(
    list(selected),
    dtype=np.int64
)

print(
    f"Working connectome: {len(working_ids):,} neurons."
)


network = connections[
    connections["pre_root_id"].isin(selected)
    &
    connections["post_root_id"].isin(selected)
].copy()

print(
    f"Working synapses: {len(network):,}"
)


id_to_index = {
    int(root_id): i
    for i, root_id in enumerate(working_ids)
}

network["pre"] = network["pre_root_id"].map(id_to_index)

network["post"] = network["post_root_id"].map(id_to_index)

network = network.dropna(
    subset=["pre", "post"]
)

network["pre"] = network["pre"].astype(np.int32)

network["post"] = network["post"].astype(np.int32)



nt_gain = {
    "ACH": EXCITATORY_GAIN,
    "GLUT": EXCITATORY_GAIN,
    "GABA": -INHIBITORY_GAIN,
    "DA": 0.45,
    "SER": 0.45,
    "OCT": 0.45,
    "UNKNOWN": 0.5,
}

network["sign"] = network["nt_type"].map(
    nt_gain
).fillna(0.5)


network["weight"] = (
    np.log1p(network["syn_count"].to_numpy())
    * network["sign"].to_numpy()
)


print("Indexing synapses...")

outgoing = {}

for row in network[
    ["pre", "post", "weight"]
].itertuples(index=False):

    pre = int(row.pre)
    post = int(row.post)
    weight = float(row.weight)

    if pre not in outgoing:
        outgoing[pre] = []

    outgoing[pre].append(
        (post, weight)
    )

print(
    f"Indexed {len(outgoing):,} presynaptic neurons."
)


neuron_metadata = {}

for row in neurons.itertuples(index=False):

    root_id = int(row.root_id)

    if root_id not in id_to_index:
        continue

    neuron_metadata[root_id] = {
        "nt": getattr(row, "nt_type", "UNKNOWN"),
        "group": getattr(row, "group", ""),
    }

cell_type_metadata = {}

for row in cell_types.itertuples(index=False):

    root_id = int(row.root_id)

    if root_id not in id_to_index:
        continue

    cell_type_metadata[root_id] = getattr(
        row,
        "primary_type",
        ""
    )

classification_metadata = {}

for row in classification.itertuples(index=False):

    root_id = int(row.root_id)

    if root_id not in id_to_index:
        continue

    classification_metadata[root_id] = {
        "flow": getattr(row, "flow", ""),
        "super_class": getattr(row, "super_class", ""),
        "class": getattr(row, "class", ""),
        "sub_class": getattr(row, "sub_class", ""),
        "side": getattr(row, "side", ""),
    }


N = len(working_ids)

activity = np.zeros(
    N,
    dtype=np.float32
)

previous_activity = np.zeros(
    N,
    dtype=np.float32
)

dopamine = 0.25
serotonin = 0.25
octopamine = 0.25

energy = 0.75
hunger = 0.25
arousal = 0.45
curiosity = 0.65

conversation_count = 0


def text_features(text):

    text = text.lower()

    features = {
        "question": "?" in text
        or bool(
            re.search(
                r"\b(what|why|how|where|when|who|are|is|can|do|does)\b",
                text
            )
        ),

        "social": bool(
            re.search(
                r"\b(hi|hello|hey|sup|you|your|name|feel|think|love|hate)\b",
                text
            )
        ),

        "threat": bool(
            re.search(
                r"\b(danger|threat|attack|kill|dead|die|scared|fear)\b",
                text
            )
        ),

        "food": bool(
            re.search(
                r"\b(food|eat|hungry|sugar|sweet|fruit|drink)\b",
                text
            )
        ),

        "math": bool(
            re.search(
                r"\d+\s*[\+\-\*/]\s*\d+",
                text
            )
        ),

        "brain": bool(
            re.search(
                r"\b(brain|neuron|neural|synapse|connectome|fafb)\b",
                text
            )
        ),

        "memory": bool(
            re.search(
                r"\b(remember|memory|earlier|before|said)\b",
                text
            )
        ),

        "curiosity": bool(
            re.search(
                r"\b(why|how|what|interesting|curious|wonder)\b",
                text
            )
        ),
    }

    return features


def inject_input(text):

    global dopamine
    global serotonin
    global octopamine
    global hunger
    global arousal
    global curiosity

    features = text_features(text)


    words = re.findall(
        r"[a-zA-Z0-9]+",
        text.lower()
    )

    if not words:
        return

    input_strength = 1.0

    for word in words:

        h = 2166136261

        for char in word:
            h ^= ord(char)
            h = (
                h * 16777619
            ) & 0xffffffff

        index = h % N

        activity[index] += (
            input_strength
        )


        for offset in (1, 7, 31, 127):

            j = (
                index + offset
            ) % N

            activity[j] += (
                input_strength * 0.35
            )


    if features["threat"]:
        arousal += 0.20
        octopamine += 0.12

    if features["food"]:
        hunger -= 0.12
        dopamine += 0.08

    if features["question"]:
        curiosity += 0.08

    if features["curiosity"]:
        curiosity += 0.10

    if features["social"]:
        serotonin += 0.04

    if features["brain"]:
        curiosity += 0.12

    hunger = np.clip(
        hunger,
        0.0,
        1.0
    )

    arousal = np.clip(
        arousal,
        0.0,
        1.0
    )

    curiosity = np.clip(
        curiosity,
        0.0,
        1.0
    )

    dopamine = np.clip(
        dopamine,
        0.0,
        1.0
    )

    serotonin = np.clip(
        serotonin,
        0.0,
        1.0
    )

    octopamine = np.clip(
        octopamine,
        0.0,
        1.0
    )



def propagate():

    global activity
    global previous_activity

    previous_activity[:] = activity

    for step in range(PROPAGATION_STEPS):

        next_activity = (
            activity * DECAY
        )

        active_indices = np.flatnonzero(
            activity > 0.08
        )

        if len(active_indices) > ACTIVE_LIMIT:

            strengths = activity[
                active_indices
            ]

            strongest = np.argpartition(
                strengths,
                -ACTIVE_LIMIT
            )[-ACTIVE_LIMIT:]

            active_indices = active_indices[
                strongest
            ]

        for pre in active_indices:

            signal = float(
                activity[pre]
            )

            if signal <= 0:
                continue

            targets = outgoing.get(
                int(pre)
            )

            if not targets:
                continue

            for post, weight in targets:

                contribution = (
                    signal
                    * weight
                    * 0.035
                )

                next_activity[post] += (
                    contribution
                )

        next_activity = np.tanh(
            next_activity
        )

        modulation = (
            0.85
            + dopamine * 0.15
            + octopamine * 0.10
            + curiosity * 0.10
        )

        next_activity *= modulation

        next_activity = np.clip(
            next_activity,
            -1.0,
            1.0
        )

        activity = next_activity.astype(
            np.float32
        )

    previous_activity[:] = activity



def brain_state():

    active = np.abs(activity) > 0.20

    active_count = int(
        active.sum()
    )

    if active_count == 0:
        mean_activity = 0.0
    else:
        mean_activity = float(
            np.mean(
                np.abs(
                    activity[active]
                )
            )
        )

    positive = float(
        np.sum(
            activity[
                activity > 0
            ]
        )
    )

    negative = float(
        np.sum(
            np.abs(
                activity[
                    activity < 0
                ]
            )
        )
    )

    return {
        "active_neurons": active_count,
        "mean_activity": mean_activity,
        "positive_activity": positive,
        "negative_activity": negative,
        "dopamine": dopamine,
        "serotonin": serotonin,
        "octopamine": octopamine,
        "energy": energy,
        "hunger": hunger,
        "arousal": arousal,
        "curiosity": curiosity,
    }



def behavioral_state():

    state = brain_state()

    arousal_value = state["arousal"]
    curiosity_value = state["curiosity"]
    hunger_value = state["hunger"]

    activity_level = state[
        "mean_activity"
    ]

    if arousal_value > 0.75:
        behavior = "highly alert"

    elif hunger_value > 0.75:
        behavior = "food-seeking"

    elif curiosity_value > 0.80:
        behavior = "exploratory"

    elif activity_level > 0.35:
        behavior = "actively processing"

    else:
        behavior = "resting"

    return behavior



def dominant_neurons(count=12):

    if not np.any(activity):
        return []

    count = min(
        count,
        len(activity)
    )

    indices = np.argpartition(
        np.abs(activity),
        -count
    )[-count:]

    indices = indices[
        np.argsort(
            np.abs(activity[indices])
        )[::-1]
    ]

    result = []

    for index in indices:

        root_id = int(
            working_ids[index]
        )

        result.append(
            {
                "root_id": root_id,
                "activity": float(
                    activity[index]
                ),
                "cell_type": cell_type_metadata.get(
                    root_id,
                    ""
                ),
            }
        )

    return result



def internal_response(user_text):

    state = brain_state()
    behavior = behavioral_state()
    dominant = dominant_neurons()

    features = text_features(
        user_text
    )

    parts = []

    parts.append(
        f"The connectome is currently {behavior}."
    )

    parts.append(
        f"{state['active_neurons']:,} neurons "
        f"are above the activity threshold in "
        f"the current working network."
    )

    if features["threat"]:
        parts.append(
            "The input drove the fly toward a high-arousal state."
        )

    if features["food"]:
        parts.append(
            "The input reduced the simulated hunger drive."
        )

    if features["question"]:
        parts.append(
            "The input produced exploratory activity."
        )

    if features["brain"]:
        parts.append(
            "The strongest activity is being evaluated "
            "against the FAFB network rather than a scripted answer."
        )

    if dominant:

        names = []

        for item in dominant[:5]:

            cell_type = item["cell_type"]

            if isinstance(cell_type, str):
                if cell_type and cell_type != "nan":
                    names.append(
                        f"{cell_type}"
                    )

        if names:
            parts.append(
                "Dominant active cell types include "
                + ", ".join(names)
                + "."
            )

    return " ".join(parts)



def render_with_ollama(user_text, internal):

    prompt = f"""
You are the language output of an experimental fly-brain
simulation.

You are NOT the brain.

The brain has already processed the user's input using a
network derived from the FAFB v783 fly connectome.

Your only task is to express the supplied internal result
naturally as a short first-person response from the fly.

Do not invent facts.
Do not add knowledge.
Do not claim that the simulation proves biological consciousness.
Do not repeat the user's question.
Do not say you are an AI.
Do not explain the prompt.
Do not turn the response into a generic chatbot answer.

USER INPUT:
{user_text}

INTERNAL RESULT:
{internal}

Write one natural response.
"""

    try:

        result = subprocess.run(
            [
                "ollama",
                "run",
                "llama3.2:3b",
                "--nowordwrap"
            ],
            input=prompt,
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=60
        )

        output = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", result.stdout).strip()

        if output:
            return output

    except Exception:
        pass

    return internal



last_output = ""


def clean_output(text):

    global last_output

    text = text.strip()

    normalized = re.sub(
        r"[^a-z0-9 ]",
        "",
        text.lower()
    )

    previous = re.sub(
        r"[^a-z0-9 ]",
        "",
        last_output.lower()
    )

    if normalized == previous:

        return (
            "Bzz. The network settled into the same state again."
        )

    last_output = text

    return text



def idle_decay():

    global energy
    global hunger
    global arousal
    global curiosity
    global dopamine
    global serotonin
    global octopamine


    activity[:] *= 0.96

    energy -= 0.001
    hunger += 0.002

    arousal *= 0.995

    dopamine *= 0.999
    serotonin *= 0.999
    octopamine *= 0.999

    energy = np.clip(
        energy,
        0.0,
        1.0
    )

    hunger = np.clip(
        hunger,
        0.0,
        1.0
    )

    arousal = np.clip(
        arousal,
        0.0,
        1.0
    )



print()
print("=" * 64)
print(" FAFB CONNECTOME FLY")
print("=" * 64)
print(
    "Real FAFB wiring is driving a persistent sparse neural state."
)
print(
    "Ollama is used only for final language rendering."
)
print()
print("Talk normally.")
print("Type 'quit' to stop.")
print()

while True:

    try:

        user = input("You: ").strip()

    except (
        KeyboardInterrupt,
        EOFError
    ):

        print()
        print("Fly: The network is going quiet.")
        break

    if not user:
        continue

    if user.lower() in {
        "quit",
        "exit"
    }:

        print()
        print(
            "Fly: The network is going quiet."
        )
        break


    conversation_count += 1

    energy -= 0.004
    hunger += 0.006

    energy = np.clip(
        energy,
        0.0,
        1.0
    )

    hunger = np.clip(
        hunger,
        0.0,
        1.0
    )


    inject_input(user)


    start = time.perf_counter()

    propagate()

    elapsed = (
        time.perf_counter()
        - start
    )


    internal = internal_response(
        user
    )


    internal += (
        f" Propagation took {elapsed:.2f} seconds."
    )


    answer = render_with_ollama(
        user,
        internal
    )

    answer = clean_output(
        answer
    )

    print()
    print("Fly:", answer)
    print()

    idle_decay()
