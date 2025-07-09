classdef RMAB < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure       matlab.ui.Figure
        TextArea_4     matlab.ui.control.TextArea
        TextArea_3     matlab.ui.control.TextArea
        TextArea_2     matlab.ui.control.TextArea
        RestartButton  matlab.ui.control.Button
        TextArea       matlab.ui.control.TextArea
        Arm3Button     matlab.ui.control.Button
        Arm2Button     matlab.ui.control.Button
        Arm1Button     matlab.ui.control.Button
        Image3         matlab.ui.control.Image
        Image2         matlab.ui.control.Image
        Image          matlab.ui.control.Image
    end

    
    properties (Access = private)
        Trial = 1  % Description
        RewardVals = load('set_1_round.mat').rounded_matrix
        Score = 0
    end
    
    methods (Access = private)
        
        function checkTrial(app)
            if app.Trial > 200
                app.TextArea.Value = sprintf('You have finished a single episode! Your score is %d \n You can start to play again by pressing restart!',app.Score);
            end
        end
        
        function update(app, arm)
            app.Score = app.Score + app.RewardVals(arm,app.Trial);
            app.TextArea.Value = sprintf('Trial number %d ! \n You choose arm %d! You get %d \n Choose again!',app.Trial, arm, app.RewardVals(arm,app.Trial));
            app.Trial = app.Trial +1;
            app.checkTrial();
        end
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            app.TextArea.Value = sprintf(['In each trial, select one slot machine to gamble. Once you choose, you will get points! \n', ...
                'Try to get as many points as you can! More points means you will get more money! You got 200 trials! \n', ...
                'The outcome of each slot machine changes with time. \n', ...
                'Choose an arm to start playing!']);
        end

        % Button pushed function: Arm1Button
        function Arm1ButtonPushed(app, event)
            arm = 1;
            if app.Trial <= 200
                app.TextArea_2.Value = num2str(app.RewardVals(arm,app.Trial));
                app.TextArea_3.Value = '';
                app.TextArea_4.Value = '';
                app.update(arm)
            end
        end

        % Callback function: TextArea
        function TextAreaValueChanging(app, event)
            changingValue = event.Value;
            
        end

        % Button pushed function: Arm2Button
        function Arm2ButtonPushed(app, event)
            arm = 2;
            if app.Trial <= 200
                app.TextArea_2.Value = '';
                app.TextArea_3.Value = num2str(app.RewardVals(arm,app.Trial));
                app.TextArea_4.Value = '';
                app.update(arm)
            end
        end

        % Button pushed function: Arm3Button
        function Arm3ButtonPushed(app, event)
            arm = 3;
            if app.Trial <= 200
                app.TextArea_2.Value = '';
                app.TextArea_3.Value = '';
                app.TextArea_4.Value = num2str(app.RewardVals(arm,app.Trial));                
                app.update(arm)
            end
        end

        % Button pushed function: RestartButton
        function RestartButtonPushed(app, event)
            app.Trial = 1;
            app.Score = 0;
            app.TextArea.Value = sprintf(['In each trial, select one slot machine to gamble. Once you choose, you will get points! \n', ...
                'Try to get as many points as you can! More points means you will get more money! You got 200 trials! \n', ...
                'The outcome of each slot machine changes with time. \n', ...
                'Choose an arm to start playing!']);
            app.TextArea_2.Value = '';
            app.TextArea_3.Value = '';
            app.TextArea_4.Value = '';
        end

        % Callback function: TextArea_2
        function TextArea_2ValueChanging(app, event)
            changingValue = event.Value;
        end

        % Value changed function: TextArea_2
        function TextArea_arm1(app, event)
            value = app.TextArea_2.Value;
        end

        % Callback function: TextArea_3
        function TextArea_3ValueChanging(app, event)
            changingValue = event.Value;
            
        end

        % Callback function: TextArea_4
        function TextArea_4ValueChanging(app, event)
            changingValue = event.Value;
            
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.AutoResizeChildren = 'off';
            app.UIFigure.Position = [100 100 640 480];
            app.UIFigure.Name = 'MATLAB App';

            % Create Image
            app.Image = uiimage(app.UIFigure);
            app.Image.Position = [71 324 100 100];
            app.Image.ImageSource = 'green_bandit_play.png';

            % Create Image2
            app.Image2 = uiimage(app.UIFigure);
            app.Image2.Position = [271 324 100 100];
            app.Image2.ImageSource = 'red_bandit_play.png';

            % Create Image3
            app.Image3 = uiimage(app.UIFigure);
            app.Image3.Position = [448 324 100 100];
            app.Image3.ImageSource = 'blue_bandit_play.png';

            % Create Arm1Button
            app.Arm1Button = uibutton(app.UIFigure, 'push');
            app.Arm1Button.ButtonPushedFcn = createCallbackFcn(app, @Arm1ButtonPushed, true);
            app.Arm1Button.Position = [71 274 100 22];
            app.Arm1Button.Text = 'Arm 1';

            % Create Arm2Button
            app.Arm2Button = uibutton(app.UIFigure, 'push');
            app.Arm2Button.ButtonPushedFcn = createCallbackFcn(app, @Arm2ButtonPushed, true);
            app.Arm2Button.Position = [257 274 100 22];
            app.Arm2Button.Text = 'Arm 2';

            % Create Arm3Button
            app.Arm3Button = uibutton(app.UIFigure, 'push');
            app.Arm3Button.ButtonPushedFcn = createCallbackFcn(app, @Arm3ButtonPushed, true);
            app.Arm3Button.Position = [435 274 100 22];
            app.Arm3Button.Text = 'Arm 3';

            % Create TextArea
            app.TextArea = uitextarea(app.UIFigure);
            app.TextArea.ValueChangingFcn = createCallbackFcn(app, @TextAreaValueChanging, true);
            app.TextArea.Position = [139 115 367 107];

            % Create RestartButton
            app.RestartButton = uibutton(app.UIFigure, 'push');
            app.RestartButton.ButtonPushedFcn = createCallbackFcn(app, @RestartButtonPushed, true);
            app.RestartButton.Position = [246 49 141 30];
            app.RestartButton.Text = 'Restart';

            % Create TextArea_2
            app.TextArea_2 = uitextarea(app.UIFigure);
            app.TextArea_2.ValueChangedFcn = createCallbackFcn(app, @TextArea_arm1, true);
            app.TextArea_2.ValueChangingFcn = createCallbackFcn(app, @TextArea_2ValueChanging, true);
            app.TextArea_2.Position = [87 362 42 24];

            % Create TextArea_3
            app.TextArea_3 = uitextarea(app.UIFigure);
            app.TextArea_3.ValueChangingFcn = createCallbackFcn(app, @TextArea_3ValueChanging, true);
            app.TextArea_3.Position = [286 362 42 24];

            % Create TextArea_4
            app.TextArea_4 = uitextarea(app.UIFigure);
            app.TextArea_4.ValueChangingFcn = createCallbackFcn(app, @TextArea_4ValueChanging, true);
            app.TextArea_4.Position = [464 362 42 24];

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = RMAB

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end