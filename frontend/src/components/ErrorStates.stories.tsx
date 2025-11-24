/**
 * Storybook stories for error state components
 */

import type { Meta, StoryObj } from '@storybook/react';
import {
  ErrorMessage,
  InlineErrorMessage,
  ReasoningUnavailableError,
  NetworkError,
  TimeoutError,
  EmptyState,
} from './ErrorStates';

const meta: Meta = {
  title: 'Components/Error States',
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
};

export default meta;

export const BasicError: StoryObj<typeof ErrorMessage> = {
  render: () => (
    <div className="w-96">
      <ErrorMessage
        error="Bir hata oluştu. Lütfen tekrar deneyin."
        onRetry={() => alert('Retry clicked')}
      />
    </div>
  ),
};

export const ErrorWithCustomTitle: StoryObj<typeof ErrorMessage> = {
  render: () => (
    <div className="w-96">
      <ErrorMessage
        error="Öneriler yüklenirken bir hata oluştu."
        onRetry={() => alert('Retry clicked')}
        title="Öneriler Yüklenemedi"
      />
    </div>
  ),
};

export const ErrorWithoutRetry: StoryObj<typeof ErrorMessage> = {
  render: () => (
    <div className="w-96">
      <ErrorMessage error="Bu işlem için yetkiniz bulunmamaktadır." />
    </div>
  ),
};

export const InlineError: StoryObj<typeof InlineErrorMessage> = {
  render: () => (
    <div className="w-96">
      <InlineErrorMessage message="Bu alan zorunludur" />
    </div>
  ),
};

export const ReasoningUnavailable: StoryObj<typeof ReasoningUnavailableError> = {
  render: () => (
    <div className="w-96">
      <ReasoningUnavailableError onClose={() => alert('Close clicked')} />
    </div>
  ),
};

export const NetworkErrorStory: StoryObj<typeof NetworkError> = {
  render: () => (
    <div className="w-96">
      <NetworkError onRetry={() => alert('Retry clicked')} />
    </div>
  ),
};

export const TimeoutErrorStory: StoryObj<typeof TimeoutError> = {
  render: () => (
    <div className="w-96">
      <TimeoutError onRetry={() => alert('Retry clicked')} />
    </div>
  ),
};

export const EmptyStateBasic: StoryObj<typeof EmptyState> = {
  render: () => (
    <div className="w-96">
      <EmptyState />
    </div>
  ),
};

export const EmptyStateWithAction: StoryObj<typeof EmptyState> = {
  render: () => (
    <div className="w-96">
      <EmptyState
        title="Henüz Favori Yok"
        message="Beğendiğiniz hediyeleri favorilere ekleyerek daha sonra kolayca bulabilirsiniz."
        actionLabel="Hediyeleri Keşfet"
        onAction={() => alert('Action clicked')}
      />
    </div>
  ),
};
