# `scenes/` — our MuJoCo scene XMLs

Every scene file this project uses. They used to live *inside* `mujoco_menagerie/`, which was
a gitignored clone — so they were untracked, had no history, and could be changed without a
diff. That cost one silent revert of the arm gains while the comment explaining them stayed
helpfully in place. They are tracked here now, and `mujoco_menagerie/` is a pinned submodule
that stays pristine.

## Layout

| file | what it is |
|---|---|
| `franka_emika_panda/scene_libero_osc.xml` | **the live scene.** LIBERO-matched Panda on torque actuators, driven by our `OSC_POSE` port |
| `franka_emika_panda/panda_libero_osc.xml` | the arm for the above; declares `meshdir` |
| `franka_emika_panda/scene_libero_hand.xml` | earlier position-actuator variant, kept for `--control-mode ik` |
| `franka_emika_panda/panda_libero_hand.xml` | the arm for the above |
| `franka_emika_panda/scene_libero.xml` | first LIBERO-alignment scene. Historical |
| `franka_emika_panda/scene_pick_place.xml` | the original pick-and-place scene. Retired DROID track |
| `franka_emika_panda/scene_playground.xml` | interactive viewer scene |
| `franka_emika_panda/panda_robotiq.xml` | **fork** of the submodule's `panda.xml`, with a Robotiq 2F-85 grafted on |
| `franka_emika_panda/scene_base.xml` | **fork** of the submodule's `scene.xml`, including the fork above |
| `franka_fr3/fr3_robotiq.xml` | FR3 + Robotiq. Retired DROID track |
| `franka_fr3/scene_droid.xml` | the DROID scene |

## The one path rule

**MuJoCo resolves `meshdir` against the directory of the top-level file passed to
`from_xml_path()`, not against the file that declares it.** That is the whole reason these
files were trapped inside the submodule: `meshdir="assets"` only worked if the top-level file
sat next to `assets/`.

Each arm file now spells out the full way back:

```xml
<compiler angle="radian" meshdir="../../mujoco_menagerie/franka_emika_panda/assets" autolimits="true"/>
```

Mesh references *within* a file are relative to `meshdir`, so the Robotiq meshes still resolve
through `../../robotiq_2f85/assets/…` unchanged. Keep the two-level `scenes/<vendor_dir>/`
nesting — that arithmetic depends on it.

## Two files are forks, not copies

`panda.xml` in the submodule had been edited in place to add a Robotiq 2F-85 gripper. A
submodule cannot carry a local edit, so that version lives here as `panda_robotiq.xml`, with
`scene_base.xml` as the matching fork of upstream `scene.xml`. Only the retired DROID scenes
need them. **The live scene does not** — `scene_libero_osc.xml` reaches the arm through
`panda_libero_osc.xml`, which is self-contained.

## Verifying a change

Never trust the XML. Compile it and read the values back:

```bash
MUJOCO_GL=egl uv run python -c "
import mujoco
m = mujoco.MjModel.from_xml_path('scenes/franka_emika_panda/scene_libero_osc.xml')
print(m.actuator_gainprm[:7])"
```

Run from the repo root — the relative paths above assume it.
