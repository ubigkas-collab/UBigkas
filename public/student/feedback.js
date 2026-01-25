

const feedbackPool = {
    needsImprovement: [
        "It seems there is some confusion with {quiz}. Let's review the core rules and try again!",
        "Don't be discouraged! {quiz} can be tricky. A bit more practice will go a long way.",
        "Focus on the fundamental concepts of {quiz}. You've got this!",
        "Practice makes perfect. Re-reading the examples for {quiz} might help clarify things.",
        "A little more focus on {quiz} will help you master it. Try reviewing the grammar cards!",
        "Mistakes are part of learning! Take another look at {quiz} and try to beat your score.",
        "You're just a few study sessions away from mastering {quiz}. Keep at it!",
        "Let's sharpen those skills in {quiz}. Persistence is the key to success!",
        "Reviewing the basic structure of {quiz} could really boost your confidence.",
        "Every expert was once a beginner. Keep practicing {quiz}!",
        "Try breaking down the {quiz} rules into smaller parts to understand them better.",
        "You're making effort, and that's what counts. Let's improve those {quiz} scores!"
    ],
    proficient: [
        "Great job! You have a solid grasp of {quiz}, but there is still room for a perfect score.",
        "You're doing well with {quiz}. Just a few more exercises and you'll be an expert!",
        "Good progress on {quiz}! Keep up the consistent effort.",
        "Your understanding of {quiz} is clear. Aim for 100% on your next attempt!",
        "Solid performance! You're very close to mastering the nuances of {quiz}.",
        "You've shown a strong handle on {quiz}. One more review could make it perfect!",
        "Impressive work! You've clearly spent time studying {quiz}.",
        "Nicely done! Your score shows you're comfortable with {quiz}.",
        "You have a good foundation in {quiz}. Now, let's aim for total mastery!",
        "You're consistently performing well in {quiz}. Keep that momentum going.",
        "Your grasp of {quiz} is commendable. Just a few minor tweaks needed!",
        "Strong results! You're definitely heading in the right direction with {quiz}."
    ],
    advanced: [
        "Excellent! You have mastered {quiz} completely.",
        "Fantastic work! Your knowledge of {quiz} is impressive.",
        "Perfect or near-perfect! You are a pro at {quiz}.",
        "Outstanding performance! You clearly understand the nuances of {quiz}.",
        "Incredible! Your mastery over {quiz} is truly top-tier.",
        "Bravo! You've tackled the most difficult parts of {quiz} with ease.",
        "You're a natural with {quiz}! Keep maintaining this level of excellence.",
        "Superb! Your hard work in studying {quiz} is clearly paying off.",
        "Top-notch performance! You've set the bar high for {quiz}.",
        "Brilliant! Your understanding of {quiz} is absolutely flawless.",
        "You have truly conquered {quiz}. Exceptional work!",
        "Expert level reached! You've demonstrated a deep understanding of {quiz}."
    ]
};

// Tracks what has been used globally across sessions
let globalShown = { needsImprovement: [], proficient: [], advanced: [] };

// Tracks what is currently on the screen to prevent duplicate comments in one list
let usedInCurrentView = [];

/**
 * Resets the "Current View" tracker. 
 * Call this at the START of your loadAssessmentScores function.
 */
export function resetViewTracker() {
    usedInCurrentView = [];
}

/**
 * Returns a varied, unique feedback string.
 */
export function generateFeedback(quizName, percentage) {
    let category;
    if (percentage < 75) {
        category = "needsImprovement";
    } else if (percentage >= 75 && percentage < 90) {
        category = "proficient";
    } else {
        category = "advanced";
    }

    const options = feedbackPool[category];
    
    // 1. Filter out what's ALREADY on the screen right now
    // 2. Filter out what was shown in previous sessions (if possible)
    let availableOptions = options.filter(opt => 
        !usedInCurrentView.includes(opt) && !globalShown[category].includes(opt)
    );

    // If we run out of unique things to say that haven't been seen in previous sessions,
    // just make sure it's not a duplicate of what's currently on the screen.
    if (availableOptions.length === 0) {
        availableOptions = options.filter(opt => !usedInCurrentView.includes(opt));
        globalShown[category] = []; // Reset global history for this category
    }
    
    // Safety check: if the pool is tiny and we still have no options, just use the pool
    if (availableOptions.length === 0) availableOptions = options;

    const randomIndex = Math.floor(Math.random() * availableOptions.length);
    const selectedFeedback = availableOptions[randomIndex];

    // Mark as used so no other quiz in this dashboard load uses it
    usedInCurrentView.push(selectedFeedback);
    globalShown[category].push(selectedFeedback);
    
    const formattedName = formatQuizName(quizName);
    return selectedFeedback.replace("{quiz}", formattedName);
}

function formatQuizName(name) {
    const names = {
        vso: "Sentence Structure",
        pronouns: "Pronouns",
        affixes: "Verbs/Affixes",
        noun: "Nouns",
        adjective: "Adjectives",
        adverb: "Adverbs"
    };
    return names[name] || name;
}