"""
Gaming-related data including game names, terms, and keywords.
This file can be easily updated to add more games and gaming terminology.
"""

# Game names and their aliases
GAME_NAMES = {
    'dota 2': ['dota', 'dota2', 'dota 2', 'defense of the ancients'],
    'valorant': ['valorant', 'valo'],
    'counter-strike': ['cs', 'csgo', 'cs:go', 'counter strike', 'counter-strike'],
    'league of legends': ['league', 'lol', 'league of legends'],
    'mortal kombat': ['mortal kombat', 'mk', 'mortal combat'],
    'fortnite': ['fortnite', 'fort', 'fn'],
    'minecraft': ['minecraft', 'mc'],
    'call of duty': ['cod', 'call of duty', 'warzone', 'modern warfare'],
    'apex legends': ['apex', 'apex legends'],
    'overwatch': ['overwatch', 'ow', 'overwatch 2', 'ow2'],
    'pubg': ['pubg', 'playerunknown', 'battlegrounds'],
    'gta': ['gta', 'grand theft auto'],
    'world of warcraft': ['wow', 'world of warcraft', 'warcraft'],
    'hearthstone': ['hearthstone', 'hs'],
    'rocket league': ['rocket league', 'rl'],
}

# Gaming-related keywords by category
GAMING_TERMS = {
    'general': {
        'game', 'gaming', 'play', 'player', 'gamer', 'gameplay', 'playthrough',
        'meta', 'patch', 'update', 'season', 'ranked', 'casual', 'competitive'
    },
    
    'moba_terms': {
        'hero', 'champion', 'lane', 'creep', 'ward', 'jungle', 'gank', 'farm',
        'mid', 'support', 'carry', 'tank', 'dps', 'heal', 'buff', 'nerf',
        'tower', 'turret', 'inhibitor', 'barracks', 'ancient', 'nexus'
    },
    
    'fps_terms': {
        'aim', 'shoot', 'recoil', 'spray', 'crosshair', 'scope', 'awp',
        'headshot', 'flick', 'peek', 'strafe', 'camp', 'rush', 'clutch',
        'frag', 'kill', 'death', 'kd', 'bomb', 'defuse', 'plant'
    },
    
    'battle_royale': {
        'drop', 'landing', 'zone', 'circle', 'storm', 'squad', 'duo', 'solo',
        'loot', 'chest', 'supply drop', 'healing', 'shield', 'armor'
    },
    
    'rpg_terms': {
        'quest', 'mission', 'level', 'xp', 'experience', 'skill tree',
        'inventory', 'gear', 'equipment', 'npc', 'boss', 'raid', 'dungeon',
        'party', 'guild', 'clan', 'alliance'
    },
    
    'hardware': {
        'controller', 'keyboard', 'mouse', 'headset', 'gpu', 'graphics card',
        'console', 'pc', 'gaming pc', 'gaming laptop', 'monitor', 'fps',
        'refresh rate', 'latency', 'ping', 'lag'
    },
    
    'mechanics': {
        'spawn', 'respawn', 'loot', 'inventory', 'skill', 'weapon', 'map',
        'damage', 'hp', 'mana', 'cooldown', 'cd', 'aoe', 'dot', 'cc',
        'combo', 'ability', 'ultimate', 'ult', 'spell', 'item'
    },
    
    'platforms': {
        'pc', 'console', 'playstation', 'ps4', 'ps5', 'xbox', 'nintendo', 'switch',
        'mobile', 'android', 'ios', 'steam', 'epic', 'origin'
    },
    
    'game_types': {
        'rpg', 'fps', 'mmorpg', 'moba', 'strategy', 'puzzle', 'arcade', 'simulator',
        'battle royale', 'br', 'rts', 'fighting game', 'racing', 'sports'
    },
    
    'esports': {
        'tournament', 'competitive', 'esports', 'stream', 'twitch', 'pro',
        'team', 'roster', 'scrim', 'practice', 'match', 'tournament',
        'league', 'championship', 'qualifier', 'finals'
    }
}

def get_all_gaming_terms():
    """Get a set of all gaming terms from all categories"""
    all_terms = set()
    for category_terms in GAMING_TERMS.values():
        all_terms.update(category_terms)
    return all_terms

def get_game_aliases():
    """Get a dictionary of all game aliases mapping to their main names"""
    aliases = {}
    for main_name, alias_list in GAME_NAMES.items():
        for alias in alias_list:
            aliases[alias] = main_name
    return aliases 