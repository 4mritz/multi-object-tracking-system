def compute_stability(tracks_history):
    id_switches = 0

    for obj_id, frames in tracks_history.items():
        prev = None
        for f in frames:
            if prev is not None and f != prev:
                id_switches += 1
            prev = f

    total = sum(len(v) for v in tracks_history.values())
    return 1 - (id_switches / (total + 1e-6))