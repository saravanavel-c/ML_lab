import streamlit as st
import gymnasium as gym
import time
import numpy as np
from taxi import train_q_learning
import re

st.title("🚕 Taxi Q-Learning Visualization")

# Sidebar controls
episodes = st.sidebar.slider("Training Episodes", 500, 5000, 2000)
speed = st.sidebar.slider("Simulation Speed", 0.05, 2.0, 1.0)

action_names = ["🔽 Move South", "🔼 Move North", "▶️ Move East", "◀️ Move West", "⬆️ Pickup", "⬇️ Dropoff"]
location_names = {0: "🔴 Red", 1: "🟢 Green", 2: "🟡 Yellow", 3: "🔵 Blue", 4: "🚕 In Taxi"}

def parse_ansi_grid(ansi_text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', ansi_text)

def decode_state(state):
    dest = state % 4
    state //= 4
    passenger = state % 5
    state //= 5
    col = state % 5
    row = state // 5
    return row, col, passenger, dest

def inject_taxi_symbol(clean_grid, taxi_row, taxi_col, picked_up):
    """Replace taxi position in grid with custom symbol"""
    lines = clean_grid.strip().split('\n')
    # Grid rows are lines 1-5 (line 0 is top border, line 6 is bottom border)
    grid_line_index = taxi_row + 1  # offset by 1 for top border

    if grid_line_index < len(lines):
        line = lines[grid_line_index]
        # Each cell is 2 chars wide in the rendered grid
        # Find the character position of the taxi column
        # Grid format: |X: X: X: X: X|
        # Each column occupies 2 characters, starting at position 1
        char_pos = 1 + taxi_col * 2

        if char_pos < len(line):
            symbol = "🚖" if picked_up else "🔍"
            # Replace that character with our symbol
            lines[grid_line_index] = line[:char_pos] + symbol + line[char_pos + 2:]

    return '\n'.join(lines)

# --- Train Button ---
if st.button("🎓 Train Model"):
    with st.spinner(f"Training over {episodes} episodes..."):
        Q = train_q_learning(episodes)
    st.session_state["Q"] = Q
    st.session_state["trained"] = True
    st.success("✅ Training Completed! You can now simulate as many times as you want.")

if st.session_state.get("trained"):
    st.info("✅ Model is trained and ready. Click below to simulate a new run.")

    # Legend
    st.markdown("**Grid Legend:** 🔍 = Taxi searching for passenger &nbsp;&nbsp; 🚖 = Taxi carrying passenger")

    if st.button("▶️ Run Simulation"):
        Q = st.session_state["Q"]

        env = gym.make("Taxi-v3", render_mode="ansi")
        state, _ = env.reset()
        done = False

        step_count = 0
        total_reward = 0
        picked_up = False

        # Decode initial state
        _, _, passenger_idx, dest_idx = decode_state(state)

        st.markdown("---")
        st.subheader("🗺️ Episode Info")
        st.markdown(f"- **Passenger Location:** {location_names[passenger_idx]}")
        st.markdown(f"- **Destination:** {location_names[dest_idx]}")
        st.markdown("---")
        st.subheader("🎬 Simulation")

        grid_placeholder = st.empty()
        info_placeholder = st.empty()
        log_placeholder = st.empty()

        step_logs = []

        while not done:
            action = np.argmax(Q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            step_count += 1

            # Detect events
            event = ""
            if action == 4 and reward != -10:
                event = f"✅ **Passenger picked up from {location_names[passenger_idx]}!**"
                picked_up = True
            elif action == 4 and reward == -10:
                event = "❌ Wrong pickup attempt!"
            elif action == 5 and reward == 20:
                event = f"🎉 **Passenger dropped at {location_names[dest_idx]}!**"
            elif action == 5 and reward == -10:
                event = "❌ Wrong dropoff attempt!"

            # Render and clean grid
            raw_grid = env.render()
            clean_grid = parse_ansi_grid(raw_grid)

            # Decode current state to get taxi position
            taxi_row, taxi_col, _, _ = decode_state(next_state)

            # Inject our custom taxi symbol
            visual_grid = inject_taxi_symbol(clean_grid, taxi_row, taxi_col, picked_up)

            grid_placeholder.markdown(f"```\n{visual_grid}\n```")

            # Status
            if done and reward == 20:
                status = "✅ Delivered!"
            elif picked_up:
                status = "🚖 Passenger on board — heading to destination"
            else:
                status = "🔍 Searching for passenger"

            info_placeholder.markdown(f"""
| Field | Value |
|---|---|
| 🔢 Step | {step_count} |
| 🎮 Action | {action_names[action]} |
| 💰 Reward | {reward} |
| 🏆 Total Reward | {total_reward} |
| 🚦 Status | {status} |
            """)

            log_entry = f"**Step {step_count}:** {action_names[action]} → Reward: `{reward}`"
            if event:
                log_entry += f"  {event}"
            step_logs.append(log_entry)
            log_placeholder.markdown("\n\n".join(step_logs))

            time.sleep(speed)
            state = next_state

        # Final result
        st.markdown("---")
        if reward == 20:
            st.success(f"🎉 SUCCESS: Passenger delivered to {location_names[dest_idx]}!")
        else:
            st.error("❌ FAILED: Could not complete the task.")

        st.write(f"**Total Steps:** {step_count}")
        st.write(f"**Final Total Reward:** {total_reward}")