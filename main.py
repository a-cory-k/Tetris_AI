from tetris import App
from ю import Bot_Trainer

app = App()
bot = Bot_Trainer(app.game)
app.run(bot_enabled=True, bot=bot)