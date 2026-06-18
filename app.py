from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, session, Response
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import datetime
import os
import cv2
import numpy as np
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = "yoga_secret_key"

# Ensure database directory exists
os.makedirs('database', exist_ok=True)

# Initialize YOLO26-Pose Model
pose_model = YOLO("yolo26n-pose.pt")

# -------------------------------------------------------------
# CORE STORAGE & STRUCTURAL SETUPS
# -------------------------------------------------------------

def get_db_connection():
    conn = sqlite3.connect('database/users.db')
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            password TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            pose TEXT,
            accuracy INTEGER,
            date TEXT
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name TEXT,
            rating INTEGER,
            comments TEXT,
            date TEXT
        )
    ''')

    conn.commit()
    conn.close()

create_tables()

# -------------------------------------------------------------
# MASTER METADATA MAPS (ALL 30 POSES UPDATED HERE)
# -------------------------------------------------------------

normal_yoga_data = [
    # === BEGINNER POSES ===
    {
        "name": "Tadasana (Mountain Pose)",
        "image": "https://tse4.mm.bing.net/th/id/OIP.9sPJ0u_HhqIbNUbvvjm3OgHaE2?w=282&h=185&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Improves posture and full-body awareness",
        "difficulty": "Beginner"
    },
    {
        "name": "Balasana (Child Pose)",
        "image": "https://tse2.mm.bing.net/th/id/OIP.v3p_A4xgnKpzdsIu5ZOfhQHaE8?w=288&h=192&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Relaxation, hip flexibility, and stress relief",
        "difficulty": "Beginner"
    },
    {
        "name": "Adho Mukha Svanasana (Downward Dog)",
        "image": "https://tse2.mm.bing.net/th/id/OIP.TqwPT_zNQYamzI9D2A8nQgHaFj?w=237&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Strengthens upper body and stretches hamstrings",
        "difficulty": "Beginner"
    },
    {
        "name": "Marjaryasana (Cat-Cow Stretch)",
        "image": "https://tse3.mm.bing.net/th/id/OIP.fGfsw4PyU_xZeEo0hvxYFwHaEK?w=302&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Warms up the spine and relieves back tension",
        "difficulty": "Beginner"
    },
    {
        "name": "Virabhadrasana I (Warrior I Pose)",
        "image": "https://tse1.mm.bing.net/th/id/OIP.4uHhz29vqzi77pF55KKmbgHaE8?w=263&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Builds leg strength and opens chest",
        "difficulty": "Beginner"
    },
    {
        "name": "Virabhadrasana II (Warrior II Pose)",
        "image": "https://tse1.mm.bing.net/th/id/OIP.cVD_GCXzb75lpcRa-4KIowHaEe?w=253&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Increases leg stamina and stretches groin",
        "difficulty": "Beginner"
    },
    {
        "name": "Vrikshasana (Tree Pose)",
        "image": "https://tse1.mm.bing.net/th/id/OIP.ulxgZpUtAgfuOa7ZJrKEjQHaE8?w=281&h=187&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Improves balance, focus, and core stability",
        "difficulty": "Beginner"
    },
    {
        "name": "Bhujangasana (Cobra Pose)",
        "image": "https://c8.alamy.com/comp/2MNK9D9/woman-practicing-cobra-asana-in-yoga-studio-bhujangasana-pose-2MNK9D9.jpg",
        "benefit": "Strengthens spine and opens chest",
        "difficulty": "Beginner"
    },
    {
        "name": "Setu Bandha Sarvangasana (Bridge Pose)",
        "image": "https://tse2.mm.bing.net/th/id/OIP.qQHPdHj01nitt0fG9y51eAHaHa?w=178&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Opens chest cavity and safely strengthens glutes",
        "difficulty": "Beginner"
    },
    {
        "name": "Savasana (Corpse Pose)",
        "image": "https://tse2.mm.bing.net/th/id/OIP._4ex-G7N_noSBRhP3xoWZgHaEA?w=265&h=183&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Calms the nervous system and lowers body tension",
        "difficulty": "Beginner"
    },

    # === INTERMEDIATE POSES ===
    {
        "name": "Bakasana (Crow Pose)",
        "image": "https://tse3.mm.bing.net/th/id/OIP.zFASSeIYdACa0yMizzNJoQHaE8?w=263&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Builds arm balance, wrist strength, and core compression",
        "difficulty": "Intermediate"
    },
    {
        "name": "Ardha Chandrasana (Half Moon Pose)",
        "image": "data:image/webp;base64,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",
        "benefit": "Strengthens lower limbs and unlocks side-body balance",
        "difficulty": "Intermediate"
    },
    {
        "name": "Urdhva Dhanurasana (Wheel Pose)",
        "image": "data:image/webp;base64,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",
        "benefit": "Deep cardiac/thoracic heart opener that tones spine",
        "difficulty": "Intermediate"
    },
    {
        "name": "Vasisthasana (Side Plank)",
        "image": "https://tse3.mm.bing.net/th/id/OIP.WNt_mHKkAjFZDzM2U_cKjgHaE8?w=238&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Strengthens lateral obliques and shoulder stabilizers",
        "difficulty": "Intermediate"
    },
    {
        "name": "Garndasana (Eagle Pose)",
        "image": "https://tse4.mm.bing.net/th/id/OIP.COF5VkbrIqum5gxigRi1FAHaE8?w=233&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Improves central focus and unlocks major skeletal joints",
        "difficulty": "Intermediate"
    },
    {
        "name": "Navasana (Boat Pose)",
        "image": "https://th.bing.com/th/id/OIP.hrzz-dU9nhGenQGTU6V1JAHaE8?w=310&h=198&c=7&rs=1&bgcl=fffffe&r=0&o=6&dpr=1.4&pid=AlgoBlockDebug",
        "benefit": "Builds supreme core compression and pancreatic stimulation",
        "difficulty": "Intermediate"
    },
    {
        "name": "Eka Pada Rajakapotasana (King Pigeon Pose)",
        "image": "https://tse2.mm.bing.net/th/id/OIP.fGdVXklSIGteuwDMDnx31wHaEK?w=292&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Deep, restorative stretch for hip flexors and glutes",
        "difficulty": "Intermediate"
    },
    {
        "name": "Ardha Pincha Mayurasana (Dolphin Pose)",
        "image": "data:image/webp;base64,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",
        "benefit": "Builds overhead shoulder strength required for inversions",
        "difficulty": "Intermediate"
    },
    {
        "name": "Utthita Parsvakonasana (Side Angle Pose)",
        "image": "data:image/webp;base64,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",
        "benefit": "Lengthens torso from heel to hand while building stamina",
        "difficulty": "Intermediate"
    },
    {
        "name": "Ustrasana (Camel Pose)",
        "image": "data:image/webp;base64,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",
        "benefit": "Stretches chest, abs, and opens back flexions",
        "difficulty": "Intermediate"
    },

    # === ADVANCED POSES ===
    {
        "name": "Adho Mukha Vrksasana (Handstand)",
        "image": "https://tse4.mm.bing.net/th/id/OIP.eEBbeNMRyb4FYBNPY9RqQgHaE8?w=251&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Ultimate check of core stability and shoulder stack mastery",
        "difficulty": "Advanced"
    },
    {
        "name": "Astavakrasana (Eight-Angle Pose)",
        "image": "https://images.squarespace-cdn.com/content/v1/5372014be4b0db8de8ce9150/1472986072012-2WAI539VPSKQPQ2BLJDT/image-asset.jpeg",
        "benefit": "Advanced lateral twisting arm balance",
        "difficulty": "Advanced"
    },
    {
        "name": "Vrischikasana (Scorpion Pose)",
        "image": "https://tse3.mm.bing.net/th/id/OIP.pC94yyimDoxA6DFUdFuTZwHaDt?w=323&h=175&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Fuses a deep forearm inversion with intense spinal flexion",
        "difficulty": "Advanced"
    },
    {
        "name": "Natarajasana (King Dancer Pose)",
        "image": "https://tse3.mm.bing.net/th/id/OIP.wcrOeRpvlhln7YCdBGVTBAHaFj?w=217&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Tests asymmetrical overhead backbends and standing stability",
        "difficulty": "Advanced"
    },
    {
        "name": "Tittibhasana (Firefly Pose)",
        "image": "https://www.theyogacollective.com/wp-content/uploads/2020/04/AdobeStock_133419034-1200x800.jpeg",
        "benefit": "Demands high hamstring flexibility and core floor lift",
        "difficulty": "Advanced"
    },
    {
        "name": "Eka Pada Rajakapotasana II (King Pigeon II)",
        "image": "data:image/webp;base64,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",
        "benefit": "Deep lunging backbend stretching the entire psoas chain",
        "difficulty": "Advanced"
    },
    {
        "name": "Parsva Bakasana Eka Pada (Side Crow Split)",
        "image": "https://tse3.mm.bing.net/th/id/OIP.y1Ee1hDSN0EN5Bqskg9_qAHaEj?w=247&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Combines dynamic rotational arm balancing with a leg split",
        "difficulty": "Advanced"
    },
    {
        "name": "Mayurasana (Peacock Pose)",
        "image": "https://tse4.mm.bing.net/th/id/OIP.Ndq7etMkY9bRRnPKvOq_NAHaEK?w=285&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Massages internal organs via forearm alignment and core lock",
        "difficulty": "Advanced"
    },
    {
        "name": "Sirsasana Urdhva Padmasana (Lotus Headstand)",
        "image": "data:image/webp;base64,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",
        "benefit": "Fuses structural headstand inversion with full lotus legs",
        "difficulty": "Advanced"
    },
    {
        "name": "Hanumanasana (Full Splits Pose)",
        "image": "https://tse3.mm.bing.net/th/id/OIP.FJijqQeAjlCNBXfDxeXjbAHaE8?w=243&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "benefit": "Ultimate elongation of front hamstrings and back hip flexors",
        "difficulty": "Advanced"
    },
    
    # Legacy fallbacks maintained for backward disease-routing maps
    {
        "name": "Baddha Konasana (Butterfly Pose)",
        "image": "https://i0.wp.com/yogamoha.com/wp-content/uploads/2018/09/Butterfly-Pose-Badhakonasana-Cobbler-Pose.jpg?w=1491&ssl=1",
        "steps": ["Sit erect with legs straight", "Bend knees and bring feet together", "Clasp feet firmly with both hands", "Gently flap knees up and down"],
        "benefits": ["Opens up hip flexors", "Relieves pelvic congestion", "Reduces pelvic area muscle stress"],
        "difficulty": "Easy"
    },
    {
        "name": "Malasana (Garland Pose)",
        "image": "https://tse3.mm.bing.net/th/id/OIP.MJsMsElRokAA2Aky7CX9SwHaHn?w=183&h=187&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Squat down completely with feet flat", "Spread thighs wider than your torso", "Press elbows against inner knees", "Keep spine straight"],
        "benefits": ["Tones abdominal wall musculature", "Improves lower back and pelvic circulation"],
        "difficulty": "Medium"
    },
    {
        "name": "Matsyasana (Fish Pose)",
        "image": "https://tse2.mm.bing.net/th/id/OIP.HtlN9PtWNkm2bIedrHf7QQHaFS?w=258&h=185&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Lie flat on back", "Slide hands under glutes", "Press forearms down and lift your chest high", "Lower top of head toward the mat"],
        "benefits": ["Stretches and opens throat area", "Stimulates local thyroid blood supply flow"],
        "difficulty": "Medium"
    }
]

pose_details = {
    # BEGINNER
    "Mountain Pose": {
        "image": "https://tse4.mm.bing.net/th/id/OIP.9sPJ0u_HhqIbNUbvvjm3OgHaE2?w=282&h=185&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Stand tall with feet together", "Ground your feet into the floor", "Relax shoulders away from ears", "Let arms hang with palms forward"],
        "benefits": ["Improves posture", "Enhances body awareness", "Aligns spinal structure"]
    },
    "Child Pose": {
        "image": "https://tse2.mm.bing.net/th/id/OIP.NHICIz0IZQKahAGmOd8_7QHaEK?w=329&h=185&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Sit on knees", "Stretch hands forward", "Lower chest", "Relax body"],
        "benefits": ["Relaxation", "Stress relief", "Improves flexibility"]
    },
    "Downward Dog": {
        "image": "https://tse2.mm.bing.net/th/id/OIP.TqwPT_zNQYamzI9D2A8nQgHaFj?w=237&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Start on hands and knees", "Tuck toes and lift hips upward", "Press shoulders flat", "Form an upside down V-shape"],
        "benefits": ["Strengthens arms", "Stretches hamstrings and calves", "Elongates spine"]
    },
    "Cat-Cow Stretch": {
        "image": "https://tse3.mm.bing.net/th/id/OIP.fGfsw4PyU_xZeEo0hvxYFwHaEK?w=302&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Start on tabletop position", "Inhale, drop belly and lift chest (Cow)", "Exhale, arch spine upward to sky (Cat)"],
        "benefits": ["Warms up back spinal disks", "Synchronizes breathing tracking patterns"]
    },
    "Warrior Pose": { # Mapping handles legacy tracking aliases securely
        "image": "https://tse1.mm.bing.net/th/id/OIP.4uHhz29vqzi77pF55KKmbgHaE8?w=263&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Stand with legs wide", "Stretch hands sideways", "Bend one knee", "Look forward"],
        "benefits": ["Improves stamina", "Strengthens legs", "Burns calories"]
    },
    "Warrior I Pose": {
        "image": "https://tse1.mm.bing.net/th/id/OIP.4uHhz29vqzi77pF55KKmbgHaE8?w=263&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Step one foot forward into a deep lunge", "Angle back foot 45 degrees outward", "Raise arms straight over head toward sky"],
        "benefits": ["Opens thoracic chest components", "Strengthens thighs and quadriceps"]
    },
    "Warrior II Pose": {
        "image": "https://tse1.mm.bing.net/th/id/OIP.cVD_GCXzb75lpcRa-4KIowHaEe?w=253&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Open torso and hips sideways from lunge", "Extend arms out parallel to floor lines", "Gaze out past your front knuckles"],
        "benefits": ["Builds deep ankle stabilization endurance", "Opens tight groins"]
    },
    "Tree Pose": {
        "image": "https://tse1.mm.bing.net/th/id/OIP.ulxgZpUtAgfuOa7ZJrKEjQHaE8?w=281&h=187&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Stand straight", "Lift one leg", "Place foot on inner thigh", "Join hands above head", "Maintain balance"],
        "benefits": ["Improves body balance", "Improves concentration", "Strengthens legs", "Enhances posture"]
    },
    "Cobra Pose": {
        "image": "https://tse2.mm.bing.net/th/id/OIP.EW0NrY_ZG_jczYHuJJuH_QHaEK?w=326&h=183&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Lie on stomach", "Keep legs straight", "Place palms near chest", "Lift upper body slowly", "Look upward"],
        "benefits": ["Strengthens spine", "Improves flexibility", "Reduces stress", "Opens chest and shoulders"]
    },
    "Bridge Pose": {
        "image": "https://tse2.mm.bing.net/th/id/OIP.qQHPdHj01nitt0fG9y51eAHaHa?w=178&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Lie down flat on back", "Bend knees and set feet flat close to hips", "Press feet down to lift pelvis high", "Clasp hands under back"],
        "benefits": ["Opens up the thoracic chest cavity", "Decompresses spinal columns safely"]
    },
    "Corpse Pose": {
        "image": "https://tse2.mm.bing.net/th/id/OIP._4ex-G7N_noSBRhP3xoWZgHaEA?w=265&h=183&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Lie flat completely on your spine", "Let feet splay open comfortably", "Turn palms facing upward, close eyes"],
        "benefits": ["Triggers parasympathetic nervous system cascades", "Lowers elevated resting heartrates"]
    },

    # INTERMEDIATE
    "Crow Pose": {
        "image": "https://tse3.mm.bing.net/th/id/OIP.zFASSeIYdACa0yMizzNJoQHaE8?w=263&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Squat low and plant hands shoulder-width apart", "Place knees directly onto upper triceps", "Lean body forward until toes lift up"],
        "benefits": ["Builds deep arm/wrist loading integrity", "Fires up upper core stabilizers"]
    },
    "Half Moon Pose": {
        "image": "data:image/webp;base64,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",
        "steps": ["Balance on one leg while tipping torso forward", "Place down-hand on floor or support block", "Stack upper hip over lower hip completely"],
        "benefits": ["Develops lateral core balance parameters", "Opens hamstrings safely"]
    },
    "Wheel Pose": {
        "image": "data:image/webp;base64,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",
        "steps": ["Lie on back, flip hands near ears over shoulders", "Press down to launch your entire body into an arch"],
        "benefits": ["Deeply expands respiratory lung fields", "Counteracts seated posture compression"]
    },
    "Side Plank": {
        "image": "https://tse3.mm.bing.net/th/id/OIP.WNt_mHKkAjFZDzM2U_cKjgHaE8?w=238&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Turn sideways from pushup stance onto one forearm/hand", "Stack feet together and send upper hand straight up"],
        "benefits": ["Isolates obliques and serratus structures", "Stabilizes shoulder rings"]
    },
    "Eagle Pose": {
        "image": "https://tse4.mm.bing.net/th/id/OIP.COF5VkbrIqum5gxigRi1FAHaE8?w=233&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Wrap one thigh high around the other standing leg", "Cross elbows together and compress forearms tight"],
        "benefits": ["Improves joint lubrication indexes via compression", "Improves central neural tracking focus"]
    },
    "Boat Pose": {
        "image": "https://th.bing.com/th/id/OIP.hrzz-dU9nhGenQGTU6V1JAHaE8?w=310&h=198&c=7&rs=1&bgcl=fffffe&r=0&o=6&dpr=1.4&pid=AlgoBlockDebug",
        "steps": ["Sit with knees bent, feet flat", "Lean torso back slightly", "Lift feet off floor extending legs to a 45-degree angle", "Extend arms forward straight"],
        "benefits": ["Stimulates pancreatic tissue function", "Builds abdominal core compression power"]
    },
    "King Pigeon Pose": {
        "image": "https://tse2.mm.bing.net/th/id/OIP.fGdVXklSIGteuwDMDnx31wHaEK?w=292&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Fold front shin forward onto mat floor", "Slide trailing leg out straight behind hips", "Square pelvic line down towards floor surface"],
        "benefits": ["Deeply targets muscular glute clusters", "Eases sciatica nerve tracking pain"]
    },
    "Dolphin Pose": {
        "image": "data:image/webp;base64,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",
        "steps": ["Drop forearms parallel onto your floor surface", "Lift hips up and walk toes inward towards chest"],
        "benefits": ["Opens rotator cuffs", "Prepares frame overhead balances safely"]
    },
    "Side Angle Pose": {
        "image": "data:image/webp;base64,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",
        "steps": ["Set deep front knee lunge stance profile", "Drop arm to floor or thigh while extending opposite arm long"],
        "benefits": ["Builds deep athletic leg endurance frameworks", "Stretches lower intercostals"]
    },
    "Camel Pose": {
        "image": "data:image/webp;base64,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",
        "steps": ["Kneel tall on shins hip-width apart", "Arch back carefully to trace hands backward to heels"],
        "benefits": ["Stretches anterior hip flexors", "Improves overall lung capacities"]
    },

    # ADVANCED
    "Handstand": {
        "image": "https://tse4.mm.bing.net/th/id/OIP.eEBbeNMRyb4FYBNPY9RqQgHaE8?w=251&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Plant hands, kick legs straight up over shoulders", "Squeeze thighs and engage core to form a line"],
        "benefits": ["Builds absolute upper body mastery", "Increases circulation and focus"]
    },
    "Eight-Angle Pose": {
        "image": "https://images.squarespace-cdn.com/content/v1/5372014be4b0db8de8ce9150/1472986072012-2WAI539VPSKQPQ2BLJDT/image-asset.jpeg",
        "steps": ["Hook leg over upper arm, cross feet at ankles", "Chop elbows to 90 degrees while projecting legs to side"],
        "benefits": ["Improves twisting core strength", "Advanced coordination index"]
    },
    "Scorpion Pose": {
        "image": "https://tse3.mm.bing.net/th/id/OIP.pC94yyimDoxA6DFUdFuTZwHaDt?w=323&h=175&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Balance upside down on forearms", "Arch spine back until feet drop near crown of head"],
        "benefits": ["Deep spinal flexibility", "Advanced neck/shoulder ring stability"]
    },
    "King Dancer Pose": {
        "image": "https://tse3.mm.bing.net/th/id/OIP.wcrOeRpvlhln7YCdBGVTBAHaFj?w=217&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Stand on one leg, grab back foot over shoulders", "Kick leg up high to form an ornamental teardrop shape"],
        "benefits": ["Deep quad opening", "Demands supreme vestibular balancing focus"]
    },
    "Firefly Pose": {
        "image": "https://www.theyogacollective.com/wp-content/uploads/2020/04/AdobeStock_133419034-1200x800.jpeg",
        "steps": ["Weave shoulders under your inner knees", "Plant hands and push floor away to lock out legs straight"],
        "benefits": ["Deep hamstring stretching", "Demands immense abdominal lifting forces"]
    },
    "King Pigeon II": {
        "image": "data:image/webp;base64,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",
        "steps": ["Lunge deep forward, bend trailing knee upwards", "Reach overhead to trap foot, dropping head back to toes"],
        "benefits": ["Deepest stretch across lumbar/psoas borders", "Thyroid mechanical activation"]
    },
    "Side Crow Split": {
        "image": "https://tse3.mm.bing.net/th/id/OIP.y1Ee1hDSN0EN5Bqskg9_qAHaEj?w=247&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Take a side crow balance frame on upper arm structure", "Split legs apart like scissors horizontally"],
        "benefits": ["Isolates outer obliques", "Improves spatial coordination metrics"]
    },
    "Peacock Pose": {
        "image": "https://tse4.mm.bing.net/th/id/OIP.Ndq7etMkY9bRRnPKvOq_NAHaEK?w=285&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Turn fingers backward, jam elbows into core belly", "Lean body forward until full frame floats level"],
        "benefits": ["Aids liver/visceral circulation flow efficiency", "Strengthens wrist bones"]
    },
    "Lotus Headstand": {
        "image": "data:image/webp;base64,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",
        "steps": ["Invert up safely into a standard bound headstand", "Cross legs into lock position upside down carefully"],
        "benefits": ["Pelvic core strength integration", "Increases vestibular equilibrium controls"]
    },
    "Full Splits Pose": {
        "image": "https://tse3.mm.bing.net/th/id/OIP.FJijqQeAjlCNBXfDxeXjbAHaE8?w=243&h=180&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Slide front heel out and back knee backward", "Lower pelvis completely flat down to floor plane lines"],
        "benefits": ["Maximizes ultimate threshold hamstring flexibility", "Lengthens pelvic girdles safely"]
    },
    
    # LEGACY MAP BACKWARD COMPATIBILITY FALLBACKS
    "Butterfly Pose": {
        "image": "https://i0.wp.com/yogamoha.com/wp-content/uploads/2018/09/Butterfly-Pose-Badhakonasana-Cobbler-Pose.jpg?w=1491&ssl=1",
        "steps": ["Sit erect with legs straight", "Bend knees and bring feet together", "Clasp feet firmly with both hands", "Gently flap knees up and down"],
        "benefits": ["Opens up hip flexors", "Relieves pelvic congestion", "Reduces pelvic area muscle stress"]
    },
    "Garland Pose": {
        "image": "https://tse3.mm.bing.net/th/id/OIP.MJsMsElRokAA2Aky7CX9SwHaHn?w=183&h=187&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Squat down completely with feet flat", "Spread thighs wider than your torso", "Press elbows against inner knees", "Keep spine straight"],
        "benefits": ["Tones abdominal wall musculature", "Improves lower back and pelvic circulation"]
    },
    "Fish Pose": {
        "image": "https://tse2.mm.bing.net/th/id/OIP.HtlN9PtWNkm2bIedrHf7QQHaFS?w=258&h=185&c=7&r=0&o=7&dpr=1.4&pid=1.7&rm=3",
        "steps": ["Lie flat on back", "Slide hands under glutes", "Press forearms down and lift your chest high", "Lower top of head toward the mat"],
        "benefits": ["Stretches and opens throat area", "Stimulates local thyroid blood supply flow"]
    },
    "Easy Pose": {
        "image": "https://tse2.mm.bing.net/th/id/OIP.5IU9QhcJN7nGHxqlZSDxWQHaHX?pid=ImgDet&w=195&h=193&c=7&dpr=1.4&o=7&rm=3",
        "steps": ["Sit cross-legged comfortably", "Keep your spine upright and shoulders relaxed", "Place palms resting facing up on your knees", "Breathe slowly"],
        "benefits": ["Lowers physiological oxygen strain", "Promotes quiet resting parasympathetic responses"]
    }
}

# -------------------------------------------------------------
# AI POSE PROCESSING MATHEMATICS ENGINE
# -------------------------------------------------------------

def calculate_joint_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def process_live_video(target_pose):
    cap = cv2.VideoCapture(0)
    
    while True:
        success, frame = cap.read()
        if not success:
            break
            
        results = pose_model(frame, verbose=False)
        detected_text = f"Searching for {target_pose}..."
        accuracy = 0

        for r in results:
            if r.keypoints and len(r.keypoints.xy) > 0:
                keypoints = r.keypoints.xy[0].cpu().numpy()
                
                if len(keypoints) > 16:
                    # Extract joint data
                    left_hip, left_knee, left_ankle = keypoints[11], keypoints[13], keypoints[15]
                    right_hip, right_knee, right_ankle = keypoints[12], keypoints[14], keypoints[16]
                    left_shoulder = keypoints[5]
                    
                    # Check if lower body keypoints are visible in the camera frame
                    lower_body_visible = (left_knee[0] > 0 and left_ankle[0] > 0 and 
                                          right_knee[0] > 0 and right_ankle[0] > 0)

                    if target_pose in ["Tree Pose", "Vrikshasana (Tree Pose)"]:
                        if not lower_body_visible:
                            detected_text = "Please step back! Show your full legs in camera."
                            accuracy = 50
                        else:
                            r_knee_angle = calculate_joint_angle(right_hip, right_knee, right_ankle)
                            l_knee_angle = calculate_joint_angle(left_hip, left_knee, left_ankle)
                            
                            if (r_knee_angle < 130 and l_knee_angle > 160) or (l_knee_angle < 130 and r_knee_angle > 160):
                                detected_text = "Tree Pose Detected Perfectly!"
                                accuracy = 95
                            else:
                                detected_text = "Tree Pose: Place your foot firmly onto your inner thigh"
                                accuracy = 70
                                
                    elif target_pose in ["Cobra Pose", "Bhujangasana (Cobra Pose)"]:
                        if left_hip[0] == 0 or left_shoulder[0] == 0:
                            detected_text = "Please position your full side profile in frame."
                            accuracy = 50
                        else:
                            if left_shoulder[1] < left_hip[1] - 50:
                                detected_text = "Cobra Pose Detected Perfectly!"
                                accuracy = 90
                            else:
                                detected_text = "Cobra Pose: Drop your shoulders and lift your chest higher"
                                accuracy = 65
                    else:
                        detected_text = f"Target: {target_pose} (Positioning...)"
                        accuracy = 60

        # Draw UI overlay bounding box
        cv2.rectangle(frame, (10, 10), (480, 85), (0, 0, 0), -1)
        cv2.putText(frame, detected_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 148), 2)
        cv2.putText(frame, f"Match Accuracy: {accuracy}%", (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n'
               b'X-Detected-Text: ' + detected_text.encode('utf-8') + b'\r\n\r\n' + 
               frame_bytes + b'\r\n')
               
    cap.release()

# -------------------------------------------------------------
# INTERACTIVE CONTROL NAVIGATION MODULES
# -------------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')

        if not username or not email or not password or not confirm_password:
            flash('All configuration registration fields are required.')
            return redirect(url_for('register'))

        if password != confirm_password:
            flash('Passwords do not match. Please re-verify entries.')
            return redirect(url_for('register'))

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email=? OR name=?", (email, username))
        existing_user = cursor.fetchone()

        if existing_user:
            conn.close()
            flash('Email or Username is already registered.')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='scrypt')
        try:
            cursor.execute(
                "INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                (username, email, hashed_password)
            )
            conn.commit()
            conn.close()
            flash('Registration successful! Please sign in below.')
            return redirect(url_for('login'))
        except Exception as e:
            conn.close()
            flash('An error occurred during account routing production. Please try again.')
            print(f"Registration Query System Error Exception Context: {e}")
            return redirect(url_for('register'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user['password'], password):
            session['user'] = user['name']  
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid Login Credentials')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/dashboard')
def dashboard():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', username=session['user'])

@app.route('/normal_yoga')
def normal_yoga():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('normal_yoga.html', yoga_data=normal_yoga_data)

@app.route('/disease_yoga')
def disease_yoga():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('disease_yoga.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    if 'user' not in session:
        return redirect(url_for('login'))
        
    disease = request.form['disease']
    yoga_data = {
        "PCOD/PCOS": [
            {"name": "Cobra Pose", "image": pose_details["Cobra Pose"]["image"], "benefits": ["Regulates pelvic blood flow", "Balances hormone levels", "Stimulates ovarian function"]},
            {"name": "Butterfly Pose", "image": pose_details["Butterfly Pose"]["image"], "benefits": ["Opens up hip flexors", "Relieves pelvic congestion", "Reduces lower body tension"]}
        ],
        "Diabetes": [
            {"name": "Boat Pose", "image": pose_details["Boat Pose"]["image"], "benefits": ["Compresses and stimulates pancreatic tissue", "Aids systemic glucose management", "Builds abdominal wall core strength"]}
        ],
        "Thyroid disorder": [
            {"name": "Cobra Pose", "image": pose_details["Cobra Pose"]["image"], "benefits": ["Massages and stimulates throat thyroid gland", "Improves flexibility in cervical neck spine", "Supports general metabolic rates"]},
            {"name": "Bridge Pose", "image": pose_details["Bridge Pose"]["image"], "benefits": ["Applies corrective physical stretch to thyroid area", "Supports endocrine alignment"]}
        ],
        "Obesity": [
            {"name": "Warrior Pose", "image": pose_details["Warrior Pose"]["image"], "benefits": ["Elevates cardiac active heart rates", "Promotes dynamic fat burning safely", "Improves target muscle mass endurance"]}
        ],
        "Hypertension": [
            {"name": "Child Pose", "image": pose_details["Child Pose"]["image"], "benefits": ["Lowers overactive fight-or-flight responses", "Calms running systemic blood flow pressures", "Reduces tension over cardiac muscle cells"]}
        ],
        "Heart disease": [
            {"name": "Easy Pose", "image": pose_details["Easy Pose"]["image"], "benefits": ["Restricts physical workload strain over ventricles", "Enhances peripheral oxygen circulation", "Promotes internal tranquility"]}
        ],
        "Stroke": [
            {"name": "Tree Pose", "image": pose_details["Tree Pose"]["image"], "benefits": ["Re-establishes critical motor neurological connections", "Restores balance focus", "Rebuilds spatial symmetry core metrics"]}
        ],
        "Asthma": [
            {"name": "Bridge Pose", "image": pose_details["Bridge Pose"]["image"], "benefits": ["Expands compressed anterior rib cages", "Improves vital capacity ceiling variables", "Combats chronic shallow breath patterns"]}
        ],
        "Back Pain": [
            {"name": "Cobra Pose", "image": pose_details["Cobra Pose"]["image"], "benefits": ["Decompresses tight intervertebral structural boundaries", "Strengthens baseline lumbar stabilizer muscles"]},
            {"name": "Child Pose", "image": pose_details["Child Pose"]["image"], "benefits": ["Gently elongates active spine muscles", "Relieves persistent lower back physical aches"]}
        ],
        "Arthritis": [
            {"name": "Child Pose", "image": pose_details["Child Pose"]["image"], "benefits": ["Stimulates articular surface synovial fluids safely", "Eases physical joint friction and morning stiffness"]}
        ],
        "Kidney disease": [
            {"name": "Cobra Pose", "image": pose_details["Cobra Pose"]["image"], "benefits": ["Massages targeted retroperitoneal spaces", "Enhances blood flow velocity to urinary structures"]}
        ],
        "Liver disorder": [
            {"name": "Boat Pose", "image": pose_details["Boat Pose"]["image"], "benefits": ["Exerts passive physical stimulation on hepatic networks", "Aids visceral filtration optimization profiles"]}
        ],
        "Cancer": [
            {"name": "Child Pose", "image": pose_details["Child Pose"]["image"], "benefits": ["Alleviates systemic mental/physical treatment fatigue", "Gently supports passive circulatory fluid drainage channels"]}
        ],
        "Stress": [
            {"name": "Tree Pose", "image": pose_details["Tree Pose"]["image"], "benefits": ["Suppresses heightened mental sensory racing behaviors", "Brings active concentration focus back to structural anchors"]},
            {"name": "Child Pose", "image": pose_details["Child Pose"]["image"], "benefits": ["Dampens overtaxed adrenal gland output spikes", "Slows down elevated chronic body cortisol presence"]}
        ],
        "Insomnia": [
            {"name": "Child Pose", "image": pose_details["Child Pose"]["image"], "benefits": ["Triggers natural relaxation cascades before rest", "Quiets neuro-muscular hyper-arousal symptoms"]}
        ]
    }
    poses = yoga_data.get(disease, [])
    return render_template('recommend.html', disease=disease, poses=poses)

# -------------------------------------------------------------
# POSE TRACKING & REAL-TIME VIDEO FEED
# -------------------------------------------------------------

@app.route('/video_feed/<pose_name>')
def video_feed(pose_name):
    return Response(process_live_video(pose_name), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/pose_detection/<pose_name>')
def pose_detection(pose_name):
    if 'user' not in session:
        return redirect(url_for('login'))

    # COMPLETE STRING ROUTING MAP FOR ALL 30 POSES
    pose_mapping = {
        # Beginner Elements
        "Tadasana (Mountain Pose)": "Mountain Pose",
        "Balasana (Child Pose)": "Child Pose",
        "Adho Mukha Svanasana (Downward Dog)": "Downward Dog",
        "Marjaryasana (Cat-Cow Stretch)": "Cat-Cow Stretch",
        "Virabhadrasana I (Warrior I Pose)": "Warrior I Pose",
        "Virabhadrasana II (Warrior II Pose)": "Warrior II Pose",
        "Vrikshasana (Tree Pose)": "Tree Pose",
        "Bhujangasana (Cobra Pose)": "Cobra Pose",
        "Setu Bandha Sarvangasana (Bridge Pose)": "Bridge Pose",
        "Savasana (Corpse Pose)": "Corpse Pose",
        
        # Intermediate Elements
        "Bakasana (Crow Pose)": "Crow Pose",
        "Ardha Chandrasana (Half Moon Pose)": "Half Moon Pose",
        "Urdhva Dhanurasana (Wheel Pose)": "Wheel Pose",
        "Vasisthasana (Side Plank)": "Side Plank",
        "Garndasana (Eagle Pose)": "Eagle Pose",
        "Navasana (Boat Pose)": "Boat Pose",
        "Eka Pada Rajakapotasana (King Pigeon Pose)": "King Pigeon Pose",
        "Ardha Pincha Mayurasana (Dolphin Pose)": "Dolphin Pose",
        "Utthita Parsvakonasana (Side Angle Pose)": "Side Angle Pose",
        "Ustrasana (Camel Pose)": "Camel Pose",
        
        # Advanced Elements
        "Adho Mukha Vrksasana (Handstand)": "Handstand",
        "Astavakrasana (Eight-Angle Pose)": "Eight-Angle Pose",
        "Vrischikasana (Scorpion Pose)": "Scorpion Pose",
        "Natarajasana (King Dancer Pose)": "King Dancer Pose",
        "Tittibhasana (Firefly Pose)": "Firefly Pose",
        "Eka Pada Rajakapotasana II (King Pigeon II)": "King Pigeon II",
        "Parsva Bakasana Eka Pada (Side Crow Split)": "Side Crow Split",
        "Mayurasana (Peacock Pose)": "Peacock Pose",
        "Sirsasana Urdhva Padmasana (Lotus Headstand)": "Lotus Headstand",
        "Hanumanasana (Full Splits Pose)": "Full Splits Pose",
        
        # Legacy Fallbacks
        "Virabhadrasana (Warrior Pose)": "Warrior Pose",
        "Baddha Konasana (Butterfly Pose)": "Butterfly Pose",
        "Malasana (Garland Pose)": "Garland Pose",
        "Matsyasana (Fish Pose)": "Fish Pose",
        "Sukhasana (Easy Pose)": "Easy Pose"
    }

    corrected_pose = pose_mapping.get(pose_name, pose_name)
    pose = pose_details.get(corrected_pose)

    if not pose:
        pose = {}

    fixed_pose = {
        "image": pose.get("image", "https://images.unsplash.com/photo-1544367567-0f2fcb009e0b"),
        "category": "Yoga Pose",
        "difficulty": pose.get("difficulty", "Standard Tier"),
        "duration": pose.get("duration", "20 - 30 Seconds"),
        "steps": pose.get("steps", ["Practice slowly and watch your breath integration details."]),
        "benefits": pose.get("benefits", ["Improves baseline somatic wellness metrics."]),
        "precautions": pose.get("precautions", ["Listen to your body structural boundaries. Use blocks if needed."])
    }

    return render_template(
        'pose_detection.html',
        pose_name=pose_name,
        pose=fixed_pose,
        accuracy=None,
        feedback=None
    )

@app.route('/ai_detection/<pose_name>')
def ai_detection(pose_name):
    if 'user' not in session:
        return redirect(url_for('login'))

    pose_mapping = {
        # Beginner Elements
        "Tadasana (Mountain Pose)": "Mountain Pose",
        "Balasana (Child Pose)": "Child Pose",
        "Adho Mukha Svanasana (Downward Dog)": "Downward Dog",
        "Marjaryasana (Cat-Cow Stretch)": "Cat-Cow Stretch",
        "Virabhadrasana I (Warrior I Pose)": "Warrior I Pose",
        "Virabhadrasana II (Warrior II Pose)": "Warrior II Pose",
        "Vrikshasana (Tree Pose)": "Tree Pose",
        "Bhujangasana (Cobra Pose)": "Cobra Pose",
        "Setu Bandha Sarvangasana (Bridge Pose)": "Bridge Pose",
        "Savasana (Corpse Pose)": "Corpse Pose",
        
        # Intermediate Elements
        "Bakasana (Crow Pose)": "Crow Pose",
        "Ardha Chandrasana (Half Moon Pose)": "Half Moon Pose",
        "Urdhva Dhanurasana (Wheel Pose)": "Wheel Pose",
        "Vasisthasana (Side Plank)": "Side Plank",
        "Garndasana (Eagle Pose)": "Eagle Pose",
        "Navasana (Boat Pose)": "Boat Pose",
        "Eka Pada Rajakapotasana (King Pigeon Pose)": "King Pigeon Pose",
        "Ardha Pincha Mayurasana (Dolphin Pose)": "Dolphin Pose",
        "Utthita Parsvakonasana (Side Angle Pose)": "Side Angle Pose",
        "Ustrasana (Camel Pose)": "Camel Pose",
        
        # Advanced Elements
        "Adho Mukha Vrksasana (Handstand)": "Handstand",
        "Astavakrasana (Eight-Angle Pose)": "Eight-Angle Pose",
        "Vrischikasana (Scorpion Pose)": "Scorpion Pose",
        "Natarajasana (King Dancer Pose)": "King Dancer Pose",
        "Tittibhasana (Firefly Pose)": "Firefly Pose",
        "Eka Pada Rajakapotasana II (King Pigeon II)": "King Pigeon II",
        "Parsva Bakasana Eka Pada (Side Crow Split)": "Side Crow Split",
        "Mayurasana (Peacock Pose)": "Peacock Pose",
        "Sirsasana Urdhva Padmasana (Lotus Headstand)": "Lotus Headstand",
        "Hanumanasana (Full Splits Pose)": "Full Splits Pose",
        
        # Legacy Fallbacks
        "Virabhadrasana (Warrior Pose)": "Warrior Pose",
        "Baddha Konasana (Butterfly Pose)": "Butterfly Pose",
        "Malasana (Garland Pose)": "Garland Pose",
        "Matsyasana (Fish Pose)": "Fish Pose",
        "Sukhasana (Easy Pose)": "Easy Pose"
    }

    corrected_pose = pose_mapping.get(pose_name, pose_name)
    pose = pose_details.get(
        corrected_pose,
        {
            "image": "https://images.unsplash.com/photo-1544367567-0f2fcb009e0b",
            "steps": ["Step setup loading details..."],
            "benefits": ["Dynamic metabolic metrics scaling."]
        }
    )

    return render_template(
        'ai_detection.html',
        pose_name=pose_name,
        pose=pose
    )

@app.route('/profile')
def profile():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('profile.html', username=session['user'])

# -------------------------------------------------------------
# PERSISTENT FEEDBACK PROCESSING LOGS
# -------------------------------------------------------------

@app.route('/feedback')
def feedback():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('feedback.html', username=session['user'])

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    category = request.form.get('category')
    rating = request.form.get('rating')
    context_tags = request.form.get('context_tags')
    feedback_text = request.form.get('feedback')
    
    print(f"Feedback Received: {category} - Stars: {rating} - Text: {feedback_text}")
    
    flash('Your feedback data has been recorded successfully. Thank you!', 'success')
    return redirect(url_for('feedback'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

# -------------------------------------------------------------
# PROGRESS SAVING ENDPOINT
# -------------------------------------------------------------

@app.route('/save_progress', methods=['POST'])
def save_progress():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "error", "message": "No data received by backend."}), 400

        pose_name = data.get('pose')
        accuracy_score = data.get('accuracy', 0)
        
        current_user = session.get('user', 'Guest')
        today_date = datetime.datetime.now().strftime("%Y-%m-%d")

        print(f"Saving Data: User={current_user}, Pose={pose_name}, Accuracy={accuracy_score}%")
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO progress (user_name, pose, accuracy, date)
            VALUES (?, ?, ?, ?)
        ''', (current_user, pose_name, int(accuracy_score), today_date))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "success",
            "message": "Progress saved successfully!"
        })
        
    except Exception as e:
        print(f"!!! CRITICAL BACKEND DATABASE ERROR !!!: {str(e)}")
        return jsonify({
            "status": "error", 
            "message": f"Database insertion failure: {str(e)}"
        }), 500

if __name__ == '__main__':
    app.run(debug=True)