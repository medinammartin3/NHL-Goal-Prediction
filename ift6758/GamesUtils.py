
class GamesUtils:

    @staticmethod
    def get_teams(game):
        teams = {}
        home_team = game['homeTeam']['abbrev'] + ' - ' + game['homeTeam']['commonName']['default']
        away_team = game['awayTeam']['abbrev'] + ' - ' + game['awayTeam']['commonName']['default']
        home_team_id = game['homeTeam']['id']
        away_team_id = game['awayTeam']['id']
        home_team = {home_team_id:(home_team, 'home'), away_team_id:(away_team, 'away')}
        teams.update(home_team)
        return teams
    
    @staticmethod
    def get_game_roaster(game):
        roaster = {}
        for player in game["rosterSpots"]:
            player_name = {player['playerId'] : (player['firstName']['default'] + " " + player['lastName']['default'], player['headshot'])}
            roaster.update(player_name)
        return roaster