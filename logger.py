class Logger:
    log_level = 1
    debug_mode = False

    def __init__(self, level, debug_mode):
        # Static on purpose
        Logger.log_level = level
        Logger.debug_mode = debug_mode

    def debug(*vars):
        if Logger.debug_mode and list(vars)[0] <= Logger.log_level:
            print('[DEBUG]', vars)
