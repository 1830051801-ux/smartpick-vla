# Example real-log schema

`example_episode.jsonl` demonstrates the import format only; its metadata marks
it as a synthetic example and it is not evidence of a real-robot run.

Each line is one timestamped step with:

- schema and episode identifiers;
- explicit frame, metre/radian/second units, and canonical action order;
- natural-language instruction and quality class;
- robot feedback in `base_link` (metres/radians);
- canonical action `(dx_m, dy_m, dz_m, dyaw_rad, gripper)`;
- optional image path, success label and metadata.

Logs in a camera frame require an explicit calibrated transform during import.
Rows with missing units/frame identity/action order, non-finite values or
non-monotonic time are rejected.
