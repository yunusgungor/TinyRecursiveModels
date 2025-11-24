import type { Meta, StoryObj } from '@storybook/react';
import { UserProfileForm } from './UserProfileForm';

const meta = {
  title: 'Components/UserProfileForm',
  component: UserProfileForm,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
  argTypes: {
    onSubmit: { action: 'form submitted' },
  },
} satisfies Meta<typeof UserProfileForm>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    isLoading: false,
  },
};

export const Loading: Story = {
  args: {
    isLoading: true,
  },
};

export const WithInitialValues: Story = {
  args: {
    isLoading: false,
    initialValues: {
      age: 35,
      hobbies: ['cooking', 'gardening'],
      relationship: 'mother',
      budget: 500,
      occasion: 'birthday',
      personality_traits: ['practical', 'eco-friendly'],
    },
  },
};

export const MinimalBudget: Story = {
  args: {
    isLoading: false,
    initialValues: {
      age: 25,
      hobbies: ['reading'],
      relationship: 'friend',
      budget: 50,
      occasion: 'birthday',
      personality_traits: [],
    },
  },
};

export const HighBudget: Story = {
  args: {
    isLoading: false,
    initialValues: {
      age: 45,
      hobbies: ['technology', 'travel', 'photography'],
      relationship: 'spouse',
      budget: 5000,
      occasion: 'anniversary',
      personality_traits: ['luxury', 'tech-savvy'],
    },
  },
};
